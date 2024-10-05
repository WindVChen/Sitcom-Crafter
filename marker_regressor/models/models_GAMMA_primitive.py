import time
import torch
import numpy as np
import tqdm
from torch import nn
from torch.nn import functional as F
from torch import optim
import json
import os

from marker_regressor.models.baseops import MLP
from marker_regressor.models.baseops import TrainOP
from marker_regressor.models.baseops import get_scheduler
from marker_regressor.models.baseops import get_body_model
from marker_regressor.models.baseops import RotConverter

import trimesh
import tqdm


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, verbose=False):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer, verbose=verbose)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch+1) * 1.0 / self.warmup
        return lr_factor

class ResNetBlock(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, n_blocks, actfun='relu'):
        super(ResNetBlock, self).__init__()

        self.in_fc = nn.Linear(in_dim, h_dim)
        self.layers = nn.ModuleList([MLP(h_dim, h_dims=(h_dim, h_dim),
                                        activation=actfun)
                                        for _ in range(n_blocks)]) # two fc layers in each MLP
        self.out_fc = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        h = self.in_fc(x)
        for layer in self.layers:
            h = layer(h)+h
        y = self.out_fc(h)
        return y


class MoshRegressor(nn.Module):
    """the body regressor in Sitcom-Crafter.

    This body regressor takes a batch of primitive markers, and produces the corresponding body parameters as well as the betas
    The body parameter vector is in the shape of (b, 103), including the translation, global_orient, body_pose, hand_pose and betas.

    Gender is assumed to be Neutral when using the body regressor.

    """
    def __init__(self, config):
        super(MoshRegressor, self).__init__()
        self.in_dim = 67*3 # the marker dim
        self.h_dim = config['h_dim']
        self.n_blocks = config['n_blocks']
        self.n_recur = config['n_recur']
        self.body_shape_dim = 10
        self.actfun = config['actfun']
        self.use_cont = config.get('use_cont', False)
        if self.use_cont:
            self.body_dim = 3 + 1*6 + 21*6 + 24
        else:
            self.body_dim = 3 + 1*3 + 21*3 + 24

        self.beta_net = ResNetBlock(self.in_dim + self.body_shape_dim,
                                self.h_dim, self.body_shape_dim, self.n_blocks // 4,
                                actfun=self.actfun)

        self.pnet = ResNetBlock(self.in_dim+self.body_dim+self.body_shape_dim,
                                self.h_dim, self.body_dim, self.n_blocks,
                                actfun=self.actfun)

    def _cont2aa(self, xb):
        transl = xb[:,:3]
        body_ori_and_pose_cont = xb[:,3:3+22*6].contiguous()
        body_ori_and_pose_aa = RotConverter.cont2aa(body_ori_and_pose_cont.view(transl.shape[0],-1,6)
                                            ).reshape(xb.shape[0],-1)
        global_orient = body_ori_and_pose_aa[:,:3]
        body_pose = body_ori_and_pose_aa[:,3:]
        left_hand_pose = xb[:,3+22*6:3+22*6+12]
        right_hand_pose = xb[:,3+22*6+12:]
        out = torch.cat([transl,global_orient,body_pose,
                            left_hand_pose,right_hand_pose],dim=-1)
        return out


    def _forward(self,
                x_ref: torch.Tensor,
                prev_transl: torch.Tensor,
                prev_glorot: torch.Tensor,
                prev_theta: torch.Tensor,
                prev_lefthandpose: torch.Tensor,
                prev_righthandpose: torch.Tensor,
                betas: torch.Tensor,
        ) -> torch.Tensor:
        """the regressor used inside of this class.
        all inputs are in torch.FloatTensor on cuda

        Args:
            - x_ref: the target markers, [b, nk, 3]
            - prev_transl: [b, 3]
            - prev_glorot: [b, 3 or 6], axis angle or cont rot
            - prev_theta: [b, 63 or 126], 21 joints rotation, axis angle or cont rot
            - prev_left_hand_pose: [b, 12], hand pose in the pca space
            - prev_right_hand_pose: [b, 12], hand pose in the pca space
            - betas: [b,10] body shape

        Returns:
            - the body parameter vector (b, 93)

        Raises:
            None
        """
        xb = torch.cat([prev_transl, prev_glorot,
                        prev_theta,
                        prev_lefthandpose,
                        prev_righthandpose],
                        dim=-1)
        xr = x_ref.reshape(-1, self.in_dim)

        for _ in range(self.n_recur):
            xb = self.pnet(torch.cat([xr, xb, betas],dim=-1)) + xb

        return xb


    def _forward_betas(self,
                x_ref: torch.Tensor,
                prev_betas: torch.Tensor, T
        ) -> torch.Tensor:
        """the regressor used inside of this class.
        all inputs are in torch.FloatTensor on cuda

        Args:
            - x_ref: the target markers, [b*t, nk, 3]
            - prev_betas: [b,10] body shape

        Returns:
            - the body betas vector (b, 10)

        Raises:
            None
        """
        xr = x_ref.reshape(-1, self.in_dim)

        for _ in range(self.n_recur):
            prev_betas = self.beta_net(torch.cat([xr, prev_betas.repeat(1, T).view(xr.shape[0], -1)],dim=-1)) + prev_betas.repeat(1, T).view(xr.shape[0], -1)
            prev_betas = prev_betas.view(-1, T, 10).mean(dim=1)

        return prev_betas


    def forward(self,
                marker_ref: torch.Tensor, B, T
        ) -> torch.Tensor:
        """the regressor forward pass
        all inputs are in torch.FloatTensor on cuda
        all inputs are from the same gender.

        Args:
            - marker_ref: the target markers, [b*t, nk, 3]

        Returns:
            - the body parameter vector (b*t, 103), rotations are in axis-angle (default) or cont6d

        Raises:
            None
        """
        ## initialize variables
        n_meshes = marker_ref.shape[0]
        batch_transl = torch.zeros(n_meshes,3).to(marker_ref.device)
        if self.use_cont:
            batch_glorot = torch.zeros(n_meshes,6).to(marker_ref.device)
            body_pose = torch.zeros(n_meshes,21*6).to(marker_ref.device)
        else:
            batch_glorot = torch.zeros(n_meshes,3).to(marker_ref.device)
            body_pose = torch.zeros(n_meshes,21*3).to(marker_ref.device)
        body_betas = torch.zeros(B, 10).to(marker_ref.device)
        left_hand_pose = torch.zeros(n_meshes,12).to(marker_ref.device)
        right_hand_pose = torch.zeros(n_meshes,12).to(marker_ref.device)
        ## forward pass
        betas = self._forward_betas(marker_ref, body_betas, T).repeat(1, T).view(B * T, 10)

        xb = self._forward(marker_ref,
                            batch_transl, batch_glorot, body_pose,
                            left_hand_pose, right_hand_pose,
                            betas)
        if self.use_cont:
            out = self._cont2aa(xb)
        else:
            out = xb

        return torch.cat([out, betas], dim=-1)


class GAMMARegressorTrainOP(TrainOP):
    def build_model(self):
        self.model = MoshRegressor(self.modelconfig)
        self.model.train()
        self.model.to(self.device)
        self.use_cont = self.modelconfig.get('use_cont', False)
        '''get body moel'''
        bm_path = self.trainconfig['body_model_path']
        self.bm = get_body_model(bm_path,
                                model_type='smplx' if self.trainconfig['is_train_smplx'] else 'smplh', gender=self.modelconfig['gender'],
                                batch_size=self.trainconfig['batch_size']*self.modelconfig['seq_len'],
                                device=self.device)
        '''get markers'''
        marker_path = self.modelconfig['marker_filepath']
        with open(os.path.join(marker_path,'SSM2.json' if self.trainconfig['is_train_smplx'] else 'SSM-smplh.json')) as f:
            markerdict = json.load(f)['markersets'][0]['indices']
        self.markers = list(markerdict.values())

    def calc_loss(self, x_ref, beta_ref, transl_ref, glob_pose_ref, body_pose_ref, xb):
        '''
        - xb has axis-angle rotations
        '''
        body_param = {}
        body_param['transl'] = xb[:,:3]
        body_param['global_orient'] = xb[:,3:6]
        body_param['body_pose'] = xb[:,6:69]
        # body_param['left_hand_pose'] = xb[:,69:81]
        # body_param['right_hand_pose'] = xb[:,81:93]
        body_param['betas'] = xb[:, 93:]

        x_pred = self.bm(return_verts=True, **body_param).vertices[:,self.markers,:]
        loss_marker = F.l1_loss(x_ref, x_pred)
        # loss_hpose = torch.mean( (xb[:,69:93])**2 )

        if self.trainconfig['is_train_smplx']:
            # For SMPLX, the pose should be similar, while betas and transl are quite different.
            # If not apply pose regularization, the model will regress +360 degree rotations, leading Neck distortions.
            loss_gpose = F.l1_loss(glob_pose_ref.view(xb[:, 3:6].shape), xb[:, 3:6])
            loss_bpose = F.l1_loss(body_pose_ref.view(xb[:, 6:69].shape), xb[:, 6:69])

            loss = loss_marker + 0.1 * (loss_gpose + loss_bpose)
            return loss, np.array([loss_marker.item(), loss_gpose.item(), loss_bpose.item()])

        else:
            loss_beta = F.l1_loss(beta_ref.repeat(1, self.modelconfig['seq_len']).view(xb[:,93:].shape), xb[:,93:])
            loss_transl = F.l1_loss(transl_ref.view(xb[:,:3].shape), xb[:,:3])
            loss_gpose = F.l1_loss(glob_pose_ref.view(xb[:,3:6].shape), xb[:,3:6])
            loss_bpose = F.l1_loss(body_pose_ref.view(xb[:,6:69].shape), xb[:,6:69])

            loss = loss_marker + loss_beta + loss_transl + loss_gpose + loss_bpose
            return loss, np.array([loss_marker.item(), loss_beta.item(), loss_transl.item(), loss_gpose.item(), loss_bpose.item()])


    def train(self, trainloader):
        self.build_model()

        starting_epoch = 0
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.trainconfig['learning_rate'])
        # scheduler = get_scheduler(optimizer, policy='lambda',
        #                             num_epochs_fix=self.trainconfig['num_epochs_fix'],
        #                             num_epochs=self.trainconfig['num_epochs'])
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.trainconfig['num_epochs'], verbose=True)

        # training main loop
        loss_names = ['MSE_MARKER', 'MSE_GPOSE', 'MSE_BPOSE'] if self.trainconfig['is_train_smplx'] else ['MSE_MARKER', 'MSE_BETAS', 'MSE_TRANSL', 'MSE_GPOSE', 'MSE_BPOSE']
        for epoch in range(starting_epoch, self.trainconfig['num_epochs']):
            epoch_losses = 0
            epoch_nsamples = 0
            stime = time.time()
            ## training subloop for each epoch
            for iid, data in enumerate(trainloader):
                B, T = data[0].shape[:2]

                marker_ref = data[0].to(self.device)
                marker_ref = marker_ref.contiguous().view([-1, self.model.in_dim])
                marker_ref = marker_ref.view(marker_ref.shape[0], -1, 3)

                beta_ref = data[1].to(self.device)
                beta_ref = beta_ref.contiguous().view([-1, 10])

                transl_ref = data[2].to(self.device)
                glob_pose_ref = data[3].to(self.device)
                body_pose_ref = data[4].to(self.device)

                # forward pass
                xb_new = self.model(marker_ref.detach(), B, T)

                optimizer.zero_grad()
                loss, losses_items = self.calc_loss(marker_ref, beta_ref, transl_ref, glob_pose_ref, body_pose_ref, xb_new)
                loss.backward(retain_graph=False)
                optimizer.step()
                epoch_losses += losses_items
                epoch_nsamples += 1
            scheduler.step()
            ## logging
            epoch_losses /= epoch_nsamples
            eps_time = time.time()-stime
            lr = optimizer.param_groups[0]['lr']
            info_str = '[epoch {:d}]:'.format(epoch+1)
            for name, val in zip(loss_names, epoch_losses):
                self.writer.add_scalar(name, val, epoch+1)
                info_str += '{}={:f}, '.format(name, val)
            info_str += 'time={:f}, lr={:f}'.format(eps_time, lr)

            self.logger.info(info_str)

            if ((1+epoch) % self.trainconfig['saving_per_X_ep']==0) :
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()
                            }, self.trainconfig['save_dir'] + "/epoch-" + str(epoch + 1) + ".ckp")

            if self.trainconfig['verbose']:
                print(info_str)

        if self.trainconfig['verbose']:
            print('[INFO]: Training completes!')
            print()



class GAMMARegressorInferOP(TrainOP):
    def build_model(self):
        self.model = MoshRegressor(self.modelconfig)
        self.model.eval()
        self.model.to(self.device)
        self.use_cont = self.modelconfig.get('use_cont', False)
        '''get body moel'''
        bm_path = self.trainconfig['body_model_path']
        self.bm = get_body_model(bm_path,
                                model_type='smplx' if self.trainconfig['is_train_smplx'] else 'smplh', gender=self.modelconfig['gender'],
                                batch_size=self.trainconfig['batch_size']*self.modelconfig['seq_len'],
                                device=self.device)
        self.bm_h = get_body_model(bm_path,
                                 model_type='smplh', gender=self.modelconfig['gender'],
                                 batch_size=self.trainconfig['batch_size'] * self.modelconfig['seq_len'],
                                 device=self.device)
        '''get markers'''
        marker_path = self.modelconfig['marker_filepath']
        with open(os.path.join(marker_path,'SSM2.json' if self.trainconfig['is_train_smplx'] else 'SSM-smplh.json')) as f:
            markerdict = json.load(f)['markersets'][0]['indices']
        self.markers = list(markerdict.values())

    @torch.no_grad()
    def infer(self, testloader):
        self.build_model()

        self.model.load_state_dict(torch.load(
            self.trainconfig["checkpoint_path"], map_location=self.device)['model_state_dict'], strict=False)

        # training main loop
        epoch_losses = 0
        epoch_nsamples = 0
        loss_names = ['MSE_MARKER', 'MSE_HPOSE']
        for iid, data in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
            B, T = data[0].shape[:2]

            marker_ref, betas_smplh, transl_smplh, glorot_smplh, thetas_smplh = data
            marker_ref = marker_ref.contiguous().view([-1, self.model.in_dim])
            marker_ref = marker_ref.view(marker_ref.shape[0], -1, 3).to(self.device)
            xb_new = self.model(marker_ref.detach(), B, T)

            loss, losses_items, smplx_pred = self.calc_loss(marker_ref, xb_new)

            body_param = {}
            body_param['transl'] = transl_smplh.squeeze(0).to(self.device)
            body_param['global_orient'] = glorot_smplh.squeeze(0).to(self.device)
            body_param['body_pose'] = thetas_smplh.squeeze(0).to(self.device)
            body_param['betas'] = betas_smplh.to(self.device)

            smplh_pred = self.bm_h(return_verts=True, **body_param)

            # visualize
            mesh1 = trimesh.Trimesh(vertices=smplx_pred.vertices[0].detach().cpu().numpy(), faces=self.bm.faces, vertex_colors=[0, 0, 255, 255])
            mesh2 = trimesh.Trimesh(vertices=smplh_pred.vertices[0].detach().cpu().numpy(), faces=self.bm_h.faces, vertex_colors=[0, 255, 0, 255])
            scene = trimesh.Scene([mesh1, mesh2])
            scene.show()
            print("loss: ", losses_items)

            epoch_losses += losses_items
            epoch_nsamples += 1

        epoch_losses /= epoch_nsamples
        info_str = ''
        for name, val in zip(loss_names, epoch_losses):
            info_str += '{}={:f}, '.format(name, val)
        print("Total dataset loss: ", info_str)

    def calc_loss(self, x_ref, xb):
        '''
        - xb has axis-angle rotations
        '''
        body_param = {}
        body_param['transl'] = xb[:,:3]
        body_param['global_orient'] = xb[:,3:6]
        body_param['body_pose'] = xb[:,6:69]
        # body_param['left_hand_pose'] = xb[:,69:81]
        # body_param['right_hand_pose'] = xb[:,81:93]
        body_param['betas'] = xb[:, 93:]

        x_pred = self.bm(return_verts=True, **body_param).vertices[:,self.markers,:]
        loss_marker = F.l1_loss(x_ref, x_pred)
        loss_hpose = torch.mean( (xb[:,69:93])**2 )
        loss = loss_marker + self.lossconfig['weight_reg_hpose']*loss_hpose
        return loss, np.array([loss_marker.item(), loss_hpose.item()]), self.bm(return_verts=True, **body_param)



