from os.path import join as pjoin

import torch
from torch.utils.data import Dataset, DataLoader
from HHInter.datasets import InterHumanDataset
from HHInter.models import *
import copy
from HHInter.datasets.evaluator_models import InterCLIP
import random
from HHInter.global_path import *
import smplx
from HHInter.custom_visualize import axis_angle_to_rot6d


class EvaluationDataset(Dataset):

    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats):
        self.device = device
        self.normalizer = MotionNormalizerTorch()
        self.model = model.to(device)
        self.model.eval()
        self.dataset = dataset
        self.max_length = dataset.max_gt_length - 1  # 299, because when transfer to InterGen, we need to calculate adjacent frames.

        # Construct sdf points
        sdf_points_extents = 3.  # InterGen dataset motion maximum extent is 6.3149.
        ceiling_height = 3.
        sdf_points_res = 128

        x = torch.linspace(-sdf_points_extents, sdf_points_extents, sdf_points_res)
        y = torch.linspace(-sdf_points_extents, sdf_points_extents, sdf_points_res)
        z = torch.linspace(-ceiling_height, ceiling_height, sdf_points_res)

        x, y, z = torch.meshgrid(x, y, z)
        self.sdf_coord = torch.stack([x, y, z], dim=-1).permute(3, 0, 1, 2).to(device)

        self.idxs = list(range(len(dataset)))
        random.shuffle(self.idxs)
        self.mm_idxs = self.idxs[:mm_num_samples]

        self.mm_num_repeats = mm_num_repeats

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        id = self.idxs[item]
        with torch.no_grad():
            data = self.dataset[id]
            name, text, motion1, motion2, motion_lens, motion_cond_length, motion_R, motion_T, motion_feet, feet_height_thresh, sdf_points = data
            # Collate the data to tensor
            name = [name]
            text = [text]
            motion1 = torch.from_numpy(motion1).type(torch.float32).unsqueeze(0)
            motion2 = torch.from_numpy(motion2).type(torch.float32).unsqueeze(0)
            motion_lens = torch.tensor([motion_lens]).long()
            motion_cond_length = torch.tensor([motion_cond_length]).long()
            motion_R = torch.from_numpy(motion_R).type(torch.float32)
            motion_T = torch.from_numpy(motion_T).type(torch.float32)
            motion_feet = torch.tensor([motion_feet]).long()
            feet_height_thresh = torch.from_numpy(feet_height_thresh).type(torch.float32)
            # Transfer sdf_points to cuda, because 3D data takes much time in the device transition. (~0.5s, but also renders 512MB for 16*4*128*128*128 cuda tensor)
            sdf_points = sdf_points.type(torch.float32).unsqueeze(0).to(self.device)

            motions = torch.cat([motion1, motion2], dim=-1)

            sdf_points = torch.cat([self.sdf_coord.unsqueeze(0).expand(motions.shape[0], -1, -1, -1, -1),
                                             sdf_points.type(torch.float32)], dim=1)

            batch = {}
            batch["text"] = list(text)
            B, T = motions.shape[0], motions.shape[1]
            batch["motion_lens"] = motion_lens
            batch["motions"] = motions.reshape(B, T, -1).type(torch.float32).to(self.device)
            batch["motion_cond_length"] = motion_cond_length.long()
            batch["sdf_points"] = sdf_points.type(torch.float32)

            batch = self.model.forward_test(batch)
            motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
            motions_output = self.normalizer.backward(motions_output)
            all_motions_output = motions_output.reshape(batch["output"].shape[0], batch["output"].shape[1], 2, 68, 3)

            motions_output = all_motions_output[:, :, :, :67, :] + all_motions_output[:, :, :, 67:, :]

            motions_output = motions_output.reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)

            motion1_marker, motion2_marker = motions_output[:, :, 0].cpu(), motions_output[:, :, 1].cpu()
            B, T = motions_output.shape[0], motions_output.shape[1]
            # +1 here is to align with GT loader output.
            if T < self.max_length + 1:
                padding_len = self.max_length + 1 - T
                D = motions_output.shape[-1]
                padding_zeros = torch.zeros((B, padding_len, D))
                motion1_marker = torch.cat((motion1_marker, padding_zeros), dim=1)
                motion2_marker = torch.cat((motion2_marker, padding_zeros), dim=1)
            assert motion1_marker.shape[1] == self.max_length + 1 == motion2_marker.shape[1]

            "====================Convert marker format to InterGen format.========================"
            # We use the SMPLH regressor here, not SMPLX regressor
            self.model.decoder.diffusion.body_regressor.load_state_dict(torch.load(
                get_smplh_body_regressor_checkpoint_path(),
                map_location=self.device)['model_state_dict'])

            bm = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplh',
                                   gender='neutral', ext='pkl',
                                   num_pca_comps=12,
                                   create_global_orient=True,
                                   create_body_pose=True,
                                   create_betas=True,
                                   create_left_hand_pose=True,
                                   create_right_hand_pose=True,
                                   create_expression=True,
                                   create_jaw_pose=True,
                                   create_leye_pose=True,
                                   create_reye_pose=True,
                                   create_transl=True,
                                   batch_size=batch["output"].shape[0] * batch["output"].shape[1]
                                   ).to(self.device)

            motion = [None, None]

            B, T = motions_output.shape[:2]
            for j in range(2):
                xb = self.model.decoder.diffusion.body_regressor(motions_output[:, :, j].reshape(-1, 67 * 3), B, T)
                # xb = torch.from_numpy(filters.gaussian_filter1d(xb.detach().cpu(), 1, axis=0, mode='nearest')).cuda()
                body_param = {}
                body_param['transl'] = xb[:, :3]
                body_param['global_orient'] = xb[:, 3:6]
                body_param['body_pose'] = xb[:, 6:69]
                # body_param['left_hand_pose'] = xb[:, 69:81]
                # body_param['right_hand_pose'] = xb[:, 81:93]
                body_param['betas'] = xb[:, 93:]

                # Smooth the params.
                # body_param['transl'] = torch.from_numpy(filters.gaussian_filter1d(body_param['transl'].detach().cpu(), 1, axis=0, mode='nearest')).cuda()

                x_pred = bm(**body_param)
                joints = x_pred.joints[:, :22, :].view(-1, 22 * 3)
                rot6d = axis_angle_to_rot6d(xb[:, 6:69]).view(-1, 21 * 6)

                motion[j] = torch.cat([joints, rot6d], dim=-1).view(batch["output"].shape[0], batch["output"].shape[1], -1).cpu().detach().numpy()

            motion1_list = []
            motion2_list = []
            for b in range(motion[0].shape[0]):
                motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion[0][b], 0.001, 0, n_joints=22)
                motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion[1][b], 0.001, 0, n_joints=22)
                r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
                angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

                xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
                relative = np.concatenate([angle, xz], axis=-1)[0]
                motion2 = rigid_transform(relative, motion2)

                motion1_list.append(motion1)
                motion2_list.append(motion2)

            motion1 = np.stack(motion1_list, axis=0)
            motion2 = np.stack(motion2_list, axis=0)

            motions_output = np.stack([motion1, motion2], axis=-2)[np.newaxis, ...].reshape(batch["output"].shape[0], batch["output"].shape[1] - 1, 2, -1)

            # motions_output[..., :22 * 3] = filters.gaussian_filter1d(motions_output[..., :22 * 3], 1, axis=0, mode='nearest')
            # motions_output[..., 22 * 3:22 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 3:22 * 6], 0.1, axis=0, mode='nearest')
            # motions_output[..., 22 * 6:22 * 6 + 21 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 6:22 * 6 + 21 * 6], 0.5, axis=0, mode='nearest')

            B,T = motions_output.shape[0], motions_output.shape[1]
            if T < self.max_length:
                padding_len = self.max_length - T
                D = motions_output.shape[-1]
                padding_zeros = np.zeros((B, padding_len, 2, D))
                motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
            assert motions_output.shape[1] == self.max_length

            sub_dict = {'motion1': motions_output[0, :,0],
                        'motion2': motions_output[0, :,1],
                        'motion_lens': motion_lens[0],
                        'text': text[0],
                        'sdf_points': sdf_points[0]}

        data = sub_dict
        motion1, motion2, motion_lens, text, sdf_points = data['motion1'], \
            data['motion2'], data['motion_lens'], data['text'], data['sdf_points']

        # -1 here because we calculate adjacent frames.
        return "generated", text, motion1, motion2, motion_lens - 1, np.zeros_like(motion_lens), np.zeros_like(motion_lens), np.zeros_like(motion_lens), \
            np.zeros_like(motion_lens), np.zeros_like(motion_lens), sdf_points.cpu(), \
            "generated", text, motion1_marker[0], motion2_marker[0], motion_lens, motion_cond_length, motion_R, motion_T, motion_feet, feet_height_thresh, sdf_points.cpu()


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.device = motion_dataset.device
        self.normalizer = MotionNormalizerTorch()
        self.model = motion_dataset.model
        self.dataset = motion_dataset.dataset
        self.max_length = motion_dataset.max_length

        self.sdf_coord = motion_dataset.sdf_coord

        self.idxs = motion_dataset.idxs
        self.mm_idxs = motion_dataset.mm_idxs

        self.mm_num_repeats = motion_dataset.mm_num_repeats

    def __len__(self):
        return len(self.mm_idxs)

    def __getitem__(self, item):
        id = self.mm_idxs[item]
        with torch.no_grad():
            data = self.dataset[id]
            name, text, motion1, motion2, motion_lens, motion_cond_length, motion_R, motion_T, motion_feet, feet_height_thresh, sdf_points = data
            # Collate the data to tensor
            name = [name]
            text = [text]
            motion1 = torch.from_numpy(motion1).type(torch.float32).unsqueeze(0)
            motion2 = torch.from_numpy(motion2).type(torch.float32).unsqueeze(0)
            motion_lens = torch.tensor([motion_lens]).long()
            motion_cond_length = torch.tensor([motion_cond_length]).long()
            motion_R = torch.from_numpy(motion_R).type(torch.float32).unsqueeze(0)
            motion_T = torch.from_numpy(motion_T).type(torch.float32).unsqueeze(0)
            motion_feet = torch.tensor([motion_feet]).long()
            feet_height_thresh = torch.from_numpy(feet_height_thresh).type(torch.float32).unsqueeze(0)
            # Transfer sdf_points to cuda, because 3D data takes much time in the device transition. (~0.5s, but also renders 512MB for 16*4*128*128*128 cuda tensor)
            sdf_points = sdf_points.type(torch.float32).unsqueeze(0).to(self.device)

            motions = torch.cat([motion1, motion2], dim=-1)

            sdf_points = torch.cat([self.sdf_coord.unsqueeze(0).expand(motions.shape[0], -1, -1, -1, -1),
                                    sdf_points.type(torch.float32)], dim=1)

            batch = {}
            batch["text"] = list(text) * self.mm_num_repeats
            B, T = motions.shape[0] * self.mm_num_repeats, motions.shape[1]
            motion_lens = motion_lens.repeat(self.mm_num_repeats)
            motion_cond_length = motion_cond_length.repeat(self.mm_num_repeats)
            motions = motions.repeat(self.mm_num_repeats, 1, 1)
            sdf_points = sdf_points.repeat(self.mm_num_repeats, 1, 1, 1, 1)

            batch["motion_lens"] = motion_lens
            batch["motions"] = motions.reshape(B, T, -1).type(torch.float32).to(self.device)
            batch["motion_cond_length"] = motion_cond_length.long()
            batch["sdf_points"] = sdf_points.type(torch.float32)

            batch = self.model.forward_test(batch)
            motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
            motions_output = self.normalizer.backward(motions_output)
            all_motions_output = motions_output.reshape(batch["output"].shape[0], batch["output"].shape[1], 2, 68, 3)

            motions_output = all_motions_output[:, :, :, :67, :] + all_motions_output[:, :, :, 67:, :]

            motions_output = motions_output.reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)

            "====================Convert marker format to InterGen format.========================"
            # We use the SMPLH regressor here, not SMPLX regressor
            self.model.decoder.diffusion.body_regressor.load_state_dict(torch.load(
                get_smplh_body_regressor_checkpoint_path(),
                map_location=self.device)['model_state_dict'])

            bm = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplh',
                              gender='neutral', ext='pkl',
                              num_pca_comps=12,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=batch["output"].shape[0] * batch["output"].shape[1]
                              ).to(self.device)

            motion = [None, None]

            B, T = motions_output.shape[:2]
            for j in range(2):
                xb = self.model.decoder.diffusion.body_regressor(motions_output[:, :, j].reshape(-1, 67 * 3), B, T)
                # xb = torch.from_numpy(filters.gaussian_filter1d(xb.detach().cpu(), 1, axis=0, mode='nearest')).cuda()
                body_param = {}
                body_param['transl'] = xb[:, :3]
                body_param['global_orient'] = xb[:, 3:6]
                body_param['body_pose'] = xb[:, 6:69]
                # body_param['left_hand_pose'] = xb[:, 69:81]
                # body_param['right_hand_pose'] = xb[:, 81:93]
                body_param['betas'] = xb[:, 93:]

                # Smooth the params.
                # body_param['transl'] = torch.from_numpy(filters.gaussian_filter1d(body_param['transl'].detach().cpu(), 1, axis=0, mode='nearest')).cuda()

                x_pred = bm(**body_param)
                joints = x_pred.joints[:, :22, :].view(-1, 22 * 3)
                rot6d = axis_angle_to_rot6d(xb[:, 6:69]).view(-1, 21 * 6)

                motion[j] = torch.cat([joints, rot6d], dim=-1).view(batch["output"].shape[0],
                                                                    batch["output"].shape[1],
                                                                    -1).cpu().detach().numpy()

            motion1_list = []
            motion2_list = []
            for b in range(motion[0].shape[0]):
                motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion[0][b], 0.001, 0, n_joints=22)
                motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion[1][b], 0.001, 0, n_joints=22)
                r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
                angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

                xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
                relative = np.concatenate([angle, xz], axis=-1)[0]
                motion2 = rigid_transform(relative, motion2)

                motion1_list.append(motion1)
                motion2_list.append(motion2)

            motion1 = np.stack(motion1_list, axis=0)
            motion2 = np.stack(motion2_list, axis=0)

            motions_output = np.stack([motion1, motion2], axis=-2)[np.newaxis, ...].reshape(
                batch["output"].shape[0], batch["output"].shape[1] - 1, 2, -1)

            # motions_output[..., :22 * 3] = filters.gaussian_filter1d(motions_output[..., :22 * 3], 1, axis=0, mode='nearest')
            # motions_output[..., 22 * 3:22 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 3:22 * 6], 0.1, axis=0, mode='nearest')
            # motions_output[..., 22 * 6:22 * 6 + 21 * 6] = filters.gaussian_filter1d(motions_output[..., 22 * 6:22 * 6 + 21 * 6], 0.5, axis=0, mode='nearest')

            B, T = motions_output.shape[0], motions_output.shape[1]
            if T < self.max_length:
                padding_len = self.max_length - T
                D = motions_output.shape[-1]
                padding_zeros = np.zeros((B, padding_len, 2, D))
                motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
            assert motions_output.shape[1] == self.max_length

            mm_sub_dict = {'mm_motions': motions_output,
                           'motion_lens': motion_lens[0],
                           'text': text[0],
                           'sdf_points': sdf_points[0]}

        data = mm_sub_dict
        mm_motions = data['mm_motions']
        motion_lens = data['motion_lens']
        sdf_points = data['sdf_points']
        mm_motions1 = mm_motions[:,:,0]
        mm_motions2 = mm_motions[:,:,1]
        text = data['text']
        motion_lens = np.array([motion_lens]*mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens - 1, \
            np.zeros_like(motion_lens), np.zeros_like(motion_lens), np.zeros_like(motion_lens), np.zeros_like(motion_lens), np.zeros_like(motion_lens), \
            sdf_points


def get_dataset_motion_loader(opt, batch_size, sdf_points_res):
    opt = copy.deepcopy(opt)
    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.NAME == 'interhuman':
        print('Loading dataset %s ...' % opt.NAME)

        dataset = InterHumanDataset(opt, sdf_points_res, is_eval=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset




def get_motion_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats):
    # Currently the configurations of two datasets are almost the same
    dataset = EvaluationDataset(model, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader, mm_motion_loader




def build_models(cfg):
    model = InterCLIP(cfg)

    checkpoint = torch.load(pjoin(os.path.dirname(__file__), '../eval_model/interclip.ckpt'),map_location="cpu")
    # checkpoint = torch.load(pjoin('checkpoints/interclip/model/5.ckpt'),map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return model


class EvaluatorModelWrapper(object):

    def __init__(self, cfg, device):

        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()

        self.body_regressor = MoshRegressor().to(device)
        self.body_regressor.load_state_dict(
            torch.load(get_smplh_body_regressor_checkpoint_path(), map_location=device)['model_state_dict'])
        self.body_regressor.eval()

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask

    @torch.no_grad()
    def convert_marker_to_intergen(self, motion1, motion2, motion_lens):
        mask = self.generate_src_mask(motion1.shape[1], motion_lens).cuda()

        self.bm = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplh',
                               gender='neutral', ext='pkl',
                               num_pca_comps=12,
                               create_global_orient=True,
                               create_body_pose=True,
                               create_betas=True,
                               create_left_hand_pose=True,
                               create_right_hand_pose=True,
                               create_expression=True,
                               create_jaw_pose=True,
                               create_leye_pose=True,
                               create_reye_pose=True,
                               create_transl=True,
                               batch_size=motion1.shape[1]
                               ).to(self.device)

        motions_output = torch.stack([motion1, motion2], dim=-2).cuda().float()
        motion = [None, None]

        B, T = motions_output.shape[0], motions_output.shape[1]

        for j in range(2):
            # The mask here is important and affect the R-Precision a lot.
            xb = self.body_regressor(motions_output[:, :, j].reshape(-1, 67 * 3), B, T, mask[..., [0]]).view(motion1.shape[0], motion1.shape[1], -1)
            # xb = torch.from_numpy(filters.gaussian_filter1d(xb.detach().cpu(), 1, axis=0, mode='nearest')).cuda()
            sub_list = []
            for sub_xb in xb:
                body_param = {}
                body_param['transl'] = sub_xb[:, :3]
                body_param['global_orient'] = sub_xb[:, 3:6]
                body_param['body_pose'] = sub_xb[:, 6:69]
                # body_param['left_hand_pose'] = sub_xb[:, 69:81]
                # body_param['right_hand_pose'] = sub_xb[:, 81:93]
                body_param['betas'] = sub_xb[:, 93:]

                # Smooth the params.
                # body_param['transl'] = torch.from_numpy(filters.gaussian_filter1d(body_param['transl'].detach().cpu(), 1, axis=0, mode='nearest')).cuda()

                x_pred = self.bm(**body_param)
                joints = x_pred.joints[:, :22, :].view(-1, 22 * 3)
                rot6d = axis_angle_to_rot6d(sub_xb[:, 6:69]).view(-1, 21 * 6)

                sub_list.append(torch.cat([joints, rot6d], dim=-1).cpu().detach())

            motion[j] = torch.stack(sub_list, dim=0).view(motion1.shape[0], motion1.shape[1], -1).numpy()

        motion1_list = []
        motion2_list = []
        for b in range(motion[0].shape[0]):
            # Motion lens mask here does not display obvious benefit, but without it, the ground height will not be right.
            motion1, root_quat_init1, root_pos_init1 = process_motion_np(motion[0][b][:motion_lens[b]], 0.001, 0, n_joints=22)
            motion2, root_quat_init2, root_pos_init2 = process_motion_np(motion[1][b][:motion_lens[b]], 0.001, 0, n_joints=22)
            r_relative = qmul_np(root_quat_init2, qinv_np(root_quat_init1))
            angle = np.arctan2(r_relative[:, 2:3], r_relative[:, 0:1])

            xz = qrot_np(root_quat_init1, root_pos_init2 - root_pos_init1)[:, [0, 2]]
            relative = np.concatenate([angle, xz], axis=-1)[0]
            motion2 = rigid_transform(relative, motion2)

            gt_length = len(motion1)
            if gt_length < len(motion[0][b]) - 1:
                padding_len = len(motion[0][b]) - 1 - gt_length
                D = motion1.shape[1]
                padding_zeros = np.zeros((padding_len, D))
                motion1 = np.concatenate((motion1, padding_zeros), axis=0)
                motion2 = np.concatenate((motion2, padding_zeros), axis=0)

            motion1_list.append(motion1)
            motion2_list.append(motion2)

        motion1 = np.stack(motion1_list, axis=0)
        motion2 = np.stack(motion2_list, axis=0)

        return torch.from_numpy(motion1), torch.from_numpy(motion2)


    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data, is_gt=False):
        with torch.no_grad():
            B, T = batch_data[2].shape[:2]
            name, text, motion1, motion2, motion_lens, _, _, _, _, _, sdf_points = batch_data
            if is_gt:
                motion1 = motion1.reshape(B, T, 68, 3)
                motion1 = motion1[:, :, :67, :] + motion1[:, :, 67:, :]
                motion1 = motion1.reshape(B, T, -1)

                motion2 = motion2.reshape(B, T, 68, 3)
                motion2 = motion2[:, :, :67, :] + motion2[:, :, 67:, :]
                motion2 = motion2.reshape(B, T, -1)
                motion1, motion2 = self.convert_marker_to_intergen(motion1[:, 1:], motion2[:, 1:], motion_lens)
                # -1 because we need to calculate the adjacent frames. This will affect the R-preceision a lot.
                motion_lens = motion_lens - 1
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

            '''Text Encoding'''
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data, is_gt=False):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens, _, _, _, _, _, sdf_points = batch_data
            B, T = batch_data[2].shape[:2]
            if is_gt:
                motion1 = motion1.reshape(B, T, 68, 3)
                motion1 = motion1[:, :, :67, :] + motion1[:, :, 67:, :]
                motion1 = motion1.reshape(B, T, -1)

                motion2 = motion2.reshape(B, T, 68, 3)
                motion2 = motion2[:, :, :67, :] + motion2[:, :, 67:, :]
                motion2 = motion2.reshape(B, T, -1)
                motion1, motion2 = self.convert_marker_to_intergen(motion1[:, 1:], motion2[:, 1:], motion_lens)
                motion_lens = motion_lens - 1
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding
