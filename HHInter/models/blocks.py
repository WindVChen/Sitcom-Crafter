import torch

from .layers import *
from torch.nn import functional as F
import torchgeometry as tgm


class TransformerBlock(nn.Module):
    def __init__(self,
                 latent_dim=512,
                 num_heads=8,
                 ff_size=1024,
                 dropout=0.,
                 cond_abl=False,
                 **kargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.cond_abl = cond_abl

        self.sa_block = VanillaSelfAttention(latent_dim, num_heads, dropout)
        self.ca_block = VanillaCrossAttention(latent_dim, latent_dim, num_heads, dropout, latent_dim)
        self.ffn = FFN(latent_dim, ff_size, dropout, latent_dim)

    def forward(self, x, y, emb=None, key_padding_mask=None):
        h1 = self.sa_block(x, emb, key_padding_mask)
        h1 = h1 + x
        h2 = self.ca_block(h1, y, emb, key_padding_mask)
        h2 = h2 + h1
        out = self.ffn(h2, emb)
        out = out + h2
        return out


class MoshRegressor(nn.Module):
    """the body regressor in Sitcom-Crafter.

    This body regressor takes a batch of primitive markers, and produces the corresponding body parameters as well as the betas
    The body parameter vector is in the shape of (b, 103), including the translation, global_orient, body_pose, hand_pose and betas.

    Gender is assumed to be Neutral when using the body regressor.

    """
    def __init__(self):
        super(MoshRegressor, self).__init__()
        self.in_dim = 67*3 # the marker dim
        self.h_dim = 128
        self.n_blocks = 10
        self.n_recur = 3
        self.body_shape_dim = 10
        self.actfun = 'relu'
        self.use_cont = True
        if self.use_cont:
            self.body_dim = 3 + 1*6 + 21*6 + 24
        else:
            self.body_dim = 3 + 1*3 + 21*3 + 24

        self.beta_net = ResNetBlock(self.in_dim + self.body_shape_dim,
                                self.h_dim, self.body_shape_dim, self.n_blocks // 4,
                                actfun=self.actfun)

        ## about the policy network
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
                prev_betas: torch.Tensor, T, mask=None
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
            prev_betas = prev_betas.view(-1, T, 10)
            if mask is None:
                prev_betas = prev_betas.mean(dim=1)
            else:
                prev_betas = (prev_betas * mask).sum(dim=1) / mask.sum(dim=1)

        return prev_betas

    def forward(self,
                marker_ref: torch.Tensor, B, T, mask=None, cur_betas=None
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
        batch_transl = torch.zeros(n_meshes, 3).to(marker_ref.device)
        if self.use_cont:
            batch_glorot = torch.zeros(n_meshes, 6).to(marker_ref.device)
            body_pose = torch.zeros(n_meshes, 21 * 6).to(marker_ref.device)
        else:
            batch_glorot = torch.zeros(n_meshes, 3).to(marker_ref.device)
            body_pose = torch.zeros(n_meshes, 21 * 3).to(marker_ref.device)
        body_betas = torch.zeros(B, 10).to(marker_ref.device)
        left_hand_pose = torch.zeros(n_meshes, 12).to(marker_ref.device)
        right_hand_pose = torch.zeros(n_meshes, 12).to(marker_ref.device)
        ## forward pass
        if cur_betas is not None:
            betas = cur_betas.repeat(1, T).view(B * T, 10)
        else:
            betas = self._forward_betas(marker_ref, body_betas, T, mask).repeat(1, T).view(B * T, 10)

        xb = self._forward(marker_ref,
                           batch_transl, batch_glorot, body_pose,
                           left_hand_pose, right_hand_pose,
                           betas)
        if self.use_cont:
            out = self._cont2aa(xb)
        else:
            out = xb

        return torch.cat([out, betas], dim=-1)


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


class RotConverter(nn.Module):
    '''
    - this class is modified from smplx/vposer
    - all functions only support data_in with [N, num_joints, D].
        -- N can be n_batch, or n_batch*n_time
    '''
    def __init__(self):
        super(RotConverter, self).__init__()

    @staticmethod
    def cont2rotmat(data_in):
        '''
        :data_in Nxnum_jointsx6
        :return: pose_matrot: Nxnum_jointsx3x3
        '''
        reshaped_input = data_in.contiguous().view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)#[b,3,3]

    @staticmethod
    def aa2cont(data_in):
        '''
        :data_in Nxnum_jointsx3
        :return: pose_matrot: Nxnum_jointsx6
        '''
        batch_size = data_in.shape[0]
        pose_body_6d = tgm.angle_axis_to_rotation_matrix(data_in.reshape(-1, 3))[:, :3, :2].contiguous().view(batch_size, -1, 6)
        return pose_body_6d


    @staticmethod
    def cont2aa(data_in):
        '''
        :data_in Nxnum_jointsx6
        :return: pose_matrot: Nxnum_jointsx3
        '''
        batch_size = data_in.shape[0]
        x_matrot_9d = RotConverter.cont2rotmat(data_in).view(batch_size,-1,9)
        x_aa = RotConverter.rotmat2aa(x_matrot_9d).contiguous().view(batch_size, -1, 3)
        return x_aa

    @staticmethod
    def rotmat2aa(data_in):
        '''
        :data_in data_in: Nxnum_jointsx9
        :return: Nxnum_jointsx3
        '''
        homogen_matrot = F.pad(data_in.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2rotmat(data_in):
        '''
        :data_in Nxnum_jointsx3
        :return: pose_matrot: Nxnum_jointsx9
        '''
        batch_size = data_in.shape[0]
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(data_in.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, -1, 9)

        return pose_body_matrot


    @staticmethod
    def vposer2rotmat6d(vposer, data_in):
        '''
        :data_in Bx32
        :return: pose_matrot: Nxnum_jointsx9
        '''
        batch_size = data_in.shape[0]
        x_pred_pose_9d = (vposer.decode(data_in).get('pose_body_matrot')).reshape(-1,1,21,3,3)#[-1, 1, n_joints, 3,3]
        x_pred_pose_6d = x_pred_pose_9d[:,:,:,:,:2].reshape([-1, data_in.shape[-1], 21*6]).permute([0,2,1])

        return x_pred_pose_6d

    @staticmethod
    def vposer2rotmat(vposer, x):
        x_pred_pose_9d = vposer.decode(x.permute([0,2,1]).reshape([-1, 32])).get('pose_body_matrot')#[-1, 1, n_joints, 9]
        x_pred_pose_9d = x_pred_pose_9d.reshape([-1, x.shape[-1], 21*9]).permute([0,2,1])

        return x_pred_pose_9d



class MLP(nn.Module):
    def __init__(self, in_dim,
                h_dims=[128,128], activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU()
        self.out_dim = h_dims[-1]
        self.layers = nn.ModuleList()
        in_dim_ = in_dim
        for h_dim in h_dims:
            self.layers.append(nn.Linear(in_dim_, h_dim))
            in_dim_ = h_dim

    def forward(self, x):
        for fc in self.layers:
            x = self.activation(fc(x))
        return x


class MAPEncoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_blocks, actfun='relu', residual=True):
        super(MAPEncoder, self).__init__()
        self.residual = residual
        self.layers = nn.ModuleList([MLP(in_dim, h_dims=(h_dim, h_dim),
                                        activation=actfun)]) # two fc layers in each MLP
        for _ in range(n_blocks - 1):
            self.layers.append(MLP(h_dim, h_dims=(h_dim, h_dim),
                activation=actfun))

    def forward(self, x):
        h = x
        for layer_idx, layer in enumerate(self.layers):
            r = h if self.residual and layer_idx > 0 else 0
            h = layer(h) + r
        y = h
        return y


class Conv3DEncode(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(Conv3DEncode, self).__init__()

        self.conv_layer1 = self._conv_layer_set(input_dims, 16)
        self.conv_layer2 = self._conv_layer_set(16, 16)
        self.conv_layer3 = self._conv_layer_set(16, output_dims)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )
        return conv_layer

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.view(out.size(0), -1)

        return out