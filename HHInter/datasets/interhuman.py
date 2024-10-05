import random

import numpy as np
import torch
import trimesh
from torch.utils import data
from tqdm import tqdm
from os.path import join as pjoin

from HHInter.utils.preprocess import *
import smplx
from scipy.spatial.transform import Rotation
from HHInter.rearrange_dataset import get_new_coordinate, get_body_model
import json
import copy
from HHInter.global_path import *
from PIL import Image, ImageDraw
import cv2


class canonicalize_world_transform():
    def __init__(self, body_model_path=get_SMPL_SMPLH_SMPLX_body_model_path()):
        self.body_model = smplx.create(body_model_path, model_type='smplh',
                                  gender='neutral', ext='pkl',
                                  num_betas=10,
                                  batch_size=1
                                  )
        self.body_model_smplx_male = smplx.create(body_model_path, model_type='smplx',
                                       gender='male', ext='pkl',
                                       num_betas=10,
                                       batch_size=1
                                       )
        self.body_model_smplx_female = smplx.create(body_model_path, model_type='smplx',
                                                  gender='female', ext='pkl',
                                                  num_betas=10,
                                                  batch_size=1
                                                  )
        with open(get_SSM_SMPL_body_marker_path()) as f:
            self.marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())
        with open(get_SSM_SMPLX_body_marker_path()) as f:
            self.marker_ssm_67_smplx = list(json.load(f)['markersets'][0]['indices'].values())

        self.feet_marker_idx = [47, 60, 55, 16, 30, 25]  # [r heel, r thumb, r little, l heel, l thumb, l little]

    def foot_detect(self, positions, height_thresh, thres=0.001):
        "Obtain contact binary values"
        velfactor, heightfactor = np.array([thres] * 6), np.array(height_thresh + 0.005)

        feet_x = (positions[1:, self.feet_marker_idx, 0] - positions[:-1, self.feet_marker_idx, 0]) ** 2
        feet_y = (positions[1:, self.feet_marker_idx, 1] - positions[:-1, self.feet_marker_idx, 1]) ** 2
        feet_z = (positions[1:, self.feet_marker_idx, 2] - positions[:-1, self.feet_marker_idx, 2]) ** 2
        feet_h = positions[:-1, self.feet_marker_idx, 2]
        feet = (((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)).astype(np.float32)

        return feet

    def transform_to_world(self, full_motion1, full_motion2, syn_scene, idx, gt_length):
        # Back to world coordinate.
        for full_motion in [full_motion1, full_motion2]:
            pelvis_original = self.body_model(betas=torch.FloatTensor(full_motion['betas']).repeat(1, 1)).joints[
                              :, 0, :].detach().cpu().numpy()

            transf_rotmat = full_motion['transf_rotmat'].reshape((3, 3))
            transf_transl = full_motion['transf_transl'].reshape((1, 3))
            full_motion['trans'] = np.matmul((full_motion['trans'][idx:idx + gt_length] + pelvis_original),
                              transf_rotmat.T) - pelvis_original + transf_transl
            r_ori = Rotation.from_rotvec(full_motion['poses'][idx:idx + gt_length, :3])
            r_new = Rotation.from_matrix(np.tile(full_motion['transf_rotmat'], [gt_length, 1, 1])) * r_ori
            full_motion['poses'] = np.concatenate([r_new.as_rotvec(), full_motion['poses'][idx:idx + gt_length, 3:]], axis=-1)

        syn_scene['vertices'] = np.einsum('ij,pj->pi', transf_rotmat, syn_scene['vertices']) + transf_transl

        return full_motion1, full_motion2, syn_scene

    def transform_to_canonical(self, full_motion1, full_motion2, syn_scene, idx, gt_length, feet_marker_A, feet_marker_B, is_interx=False):
        # Renew canonical coordinate.
        transf_rotmat, transf_transl = full_motion1['transf_rotmat'][idx], full_motion1['transf_transl'][idx]

        feet_A = self.foot_detect(full_motion1['marker_ssm2_67'][idx:idx + gt_length + 1], height_thresh=feet_marker_A)
        feet_B = self.foot_detect(full_motion2['marker_ssm2_67'][idx:idx + gt_length + 1], height_thresh=feet_marker_B)

        for full_motion in [full_motion1, full_motion2]:
            transl = full_motion['trans'][idx:idx + gt_length]
            pose = full_motion['poses'][idx:idx + gt_length]
            betas = full_motion['betas']
            joints = full_motion['joints'][idx:idx + gt_length]
            markers_67 = full_motion['marker_ssm2_67'][idx:idx + gt_length]

            ## perform transformation from the world coordinate to the amass coordinate
            ### calibrate offset
            # InterGen has only neutral, while interX has only male and female.
            body_model = self.body_model
            if is_interx:
                if full_motion['gender'] == 'male':
                    body_model = self.body_model_smplx_male
                elif full_motion['gender'] == 'female':
                    body_model = self.body_model_smplx_female

            delta_T = body_model(betas=torch.FloatTensor(betas).repeat(1, 1)).joints[
                              :, 0, :].detach().cpu().numpy()
            ### get new global_orient
            global_ori = Rotation.from_rotvec(pose[:, :3]).as_matrix()  # to [t,3,3] rotation mat
            global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat.T, global_ori)
            pose[:, :3] = Rotation.from_matrix(global_ori_new).as_rotvec()
            ### get new transl
            transl = np.einsum('ij,tj->ti', transf_rotmat.T, transl + delta_T - transf_transl) - delta_T
            full_motion['transf_rotmat'] = transf_rotmat
            full_motion['transf_transl'] = transf_transl
            full_motion['trans'] = transl
            full_motion['poses'] = pose

            ### extract joints and markers
            joints = np.einsum('ij,bpj->bpi', transf_rotmat.T, joints - transf_transl)
            markers_67 = np.einsum('ij,bpj->bpi', transf_rotmat.T, markers_67 - transf_transl)
            full_motion['joints'] = joints
            full_motion['marker_ssm2_67'] = markers_67

        syn_scene['vertices'] = np.einsum('ij,pj->pi', transf_rotmat.T, syn_scene['vertices'] - transf_transl)
        return full_motion1, full_motion2, syn_scene, feet_A, feet_B


def random_object(sdf_region, floor_height):
    height = sdf_region.shape[2]
    im_shape = sdf_region.shape[:2]

    object_number = random.randint(0, 10)

    global_mask = np.array(Image.new("L", im_shape, 0))

    for _ in range(object_number):
        mask = Image.new("L", im_shape, 0)

        # show mask.
        # mask.show(title='1')

        draw = ImageDraw.Draw(mask)
        size = (random.randint(int(im_shape[0] * 0.1), int(im_shape[0] * 0.5)),
                random.randint(int(im_shape[0] * 0.1), int(im_shape[1] * 0.5)))

        limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
        center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
        draw_type = random.randint(0, 1)
        if draw_type == 0:
            draw.rectangle(
                (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
                fill=255,
            )
        else:
            draw.ellipse(
                (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
                fill=255,
            )

        # show mask.
        # mask.show(title='2')

        # Rotate the masked region.
        angle = random.randint(0, 360)
        mask = mask.rotate(angle, fillcolor=0)

        # show mask.
        # mask.show(title='3')

        # Random height.
        obj_height = random.randint(floor_height, height - 1)

        # Set sdf_region value to -1 for the region of the mask and below the obj_height.
        mask_np = np.array(mask)
        for i in range(im_shape[0]):
            for j in range(im_shape[1]):
                if mask_np[i, j] == 255 and global_mask[i, j] == 0:
                    sdf_region[i, j, :obj_height] = -1

        global_mask = np.maximum(global_mask, mask_np)

    # cv2.imshow("global_mask", global_mask)

    return sdf_region


class InterHumanDataset(data.Dataset):
    def __init__(self, opt, sdf_points_res, is_eval=False):
        self.opt = opt
        self.cond_length = 1
        self.max_gt_length = 300
        # A different min_gt_length (e.g., 40) affects the eval result a lot (~0.05 points).
        self.min_gt_length = 14
        # if is_eval:
        #     self.min_gt_length = 14
        # else:
        #     self.min_gt_length = 40
        self.sdf_points_extents = 3.  # InterGen dataset motion maximum extent is 6.3149. For per 300 frames, only <50 clips exceed 5.
        self.ceiling_height = 3.  # For per 300 frames, no clips exceed 3.
        self.use_interx = opt.USE_INTERX
        if self.use_interx:
            print("Loading additional INTER-X dataset...")

        """ Per 300 frames extents [3, 10], three dimension and block from 0~1 to 9~10.
        [[2.6100e+02 4.5990e+03 6.4570e+03 3.1720e+03 5.6000e+02 7.3000e+01 2.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00]
         [1.2990e+03 6.4860e+03 4.4730e+03 2.5160e+03 3.2100e+02 2.9000e+01 0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00]
         [3.0000e+00 1.3888e+04 1.2330e+03 0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00 0.0000e+00]]
        """

        self.sdf_points_res = sdf_points_res

        self.max_length = self.cond_length + self.max_gt_length
        self.min_length = self.cond_length + self.min_gt_length

        self.data_list = []
        self.motion_dict = {}

        self.cache = opt.CACHE
        self.coord_transform = canonicalize_world_transform()

        "For getting the floor position"
        self.smplh_model = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplh', gender='neutral', ext='pkl',
                              num_betas=10, batch_size=1)
        self.smplx_model_male = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplx', gender='male',
                                        ext='pkl', num_betas=10, batch_size=1)
        self.smplx_model_female = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplx', gender='female',
                                             ext='pkl', num_betas=10, batch_size=1)

        with open(get_SSM_SMPL_body_marker_path()) as f:
            self.marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())
        with open(get_SSM_SMPLX_body_marker_path()) as f:
            self.marker_ssm_67_smplx = list(json.load(f)['markersets'][0]['indices'].values())

        ignore_list = []
        try:
            ignore_list = open(os.path.join(opt.DATA_ROOT, "ignore_list.txt"), "r").readlines()
        except Exception as e:
            print(e)
        data_list = []
        if self.opt.MODE == "train":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "train.txt"), "r").readlines()
                if self.use_interx:
                    data_list += open(os.path.join(opt.DATA_INTERX_ROOT, "train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "val":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "val.txt"), "r").readlines()
                if self.use_interx:
                    data_list += open(os.path.join(opt.DATA_INTERX_ROOT, "val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.opt.MODE == "test":
            try:
                data_list = open(os.path.join(opt.DATA_ROOT, "test.txt"), "r").readlines()
                if self.use_interx:
                    data_list += open(os.path.join(opt.DATA_INTERX_ROOT, "test.txt"), "r").readlines()
            except Exception as e:
                print(e)

        index = 0
        data_paths = [opt.DATA_ROOT] if not self.use_interx else [opt.DATA_ROOT, opt.DATA_INTERX_ROOT]
        for data_path in data_paths:
            for root, dirs, files in os.walk(pjoin(data_path)):
                # if index > 100: break
                for file in tqdm(files):
                    # if index > 100: break
                    if file.endswith(".npz") and "person1" in root:
                        # if int(os.path.basename(file)[:-4].replace("_swap", "")) != 4944:
                        #     continue
                        if opt.MODE != "train":
                            if "_swap" in file:
                                continue
                        motion_name = file.split(".")[0]
                        if file.split(".")[0]+"\n" in ignore_list:
                            print("ignore: ", file)
                            continue

                        "Consider both raw and swap motions"
                        if file.split(".")[0].split("_")[0]+"\n" not in data_list:
                            continue
                        file_path_person1 = pjoin(root, file)
                        file_path_person2 = pjoin(root.replace("person1", "person2"), file)
                        file_synthetic_scene = pjoin(root.replace("person1", "synthetic_scene"), file)
                        if data_path == opt.DATA_ROOT:
                            text_path = file_path_person1.replace(
                                "motions_customized_intergen_convert_fps30" if is_eval else "motions_customized_intergen_convert_fps30",
                                "annots").replace("person1", "").replace("_swap", "").replace("npz", "txt")
                        else:
                            text_path = file_path_person1.replace(
                                "motions_customized_fps30" if is_eval else "motions_customized_fps30",
                                "annots").replace("person1", "").replace("_swap", "").replace("npz", "txt")

                        with open(text_path, "r") as f:
                            texts = f.readlines()
                        texts = [item.replace("\n", "") for item in texts]

                        motion1 = load_motion(file_path_person1, self.min_length)
                        if motion1 is None:
                            continue

                        if self.cache:
                            motion2 = load_motion(file_path_person2, self.min_length)
                            syn_scene = load_scene(file_synthetic_scene)
                            self.motion_dict[index] = [motion1, motion2, syn_scene]
                        else:
                            self.motion_dict[index] = [file_path_person1, file_path_person2, file_synthetic_scene]

                        self.data_list.append({
                            "name": motion_name,
                            "motion_id": index,
                            "texts":texts
                        })

                        index += 1

        print("total dataset: ", len(self.data_list))

    def real_len(self):
        return len(self.data_list)

    def sign(self, p1, p2, p3):
        return (p1[:, :, 0] - p3[:, :, 0]) * (p2[:, :, 1] - p3[:, :, 1]) - (p2[:, :, 0] - p3[:, :, 0]) * (
                    p1[:, :, 1] - p3[:, :, 1])

    def __len__(self):
        return self.real_len()*1

    def __getitem__(self, item):
        name, text, full_motion1, full_motion2, syn_scene, idx, sample_length, length = self.sample_data(item)

        "Frist revise gt_length to ensure the motion within the sdf region."
        "Eg, for fps30 eval set, there will be ~100 clips being replaced. While for the commented strategy, " \
        "the following clips will be replaced by another clip: " \
        "[142 3925 3919 4343 3938 4378 3924 4447 4195 4346 4595 4042 4372 141 944 3920 140 685 4200]"
        sdf_region = np.array([self.sdf_points_extents, self.sdf_points_extents, self.ceiling_height])
        num_times = 0
        while True:
            num_times += 1
            markers_67_A = full_motion1['marker_ssm2_67'][idx:idx + sample_length-1]
            markers_67_B = full_motion2['marker_ssm2_67'][idx:idx + sample_length-1]

            # Stack A and B
            markers_67_stack = np.concatenate([markers_67_A, markers_67_B], axis=0)

            if "_swap" not in name:
                extent_max = np.max(markers_67_stack, axis=(0,1)) - full_motion1['joints'][idx][0]
                extent_min = np.min(markers_67_stack, axis=(0,1)) - full_motion1['joints'][idx][0]
            else:
                extent_max = np.max(markers_67_stack, axis=(0,1)) - full_motion2['joints'][idx][0]
                extent_min = np.min(markers_67_stack, axis=(0,1)) - full_motion2['joints'][idx][0]

            if np.any(extent_max > sdf_region) or np.any(extent_min < -sdf_region):
                # sample_length -= 10
                pass
            else:
                break

            # if sample_length < self.min_length:
            #     # Judge if there are other sub clips to sample.
            if length > self.max_length + 1:
                idx = random.choice(list(range(0, length - (self.max_length + 1), 1)))
                sample_length = self.max_length + 1
            else:
                "The commented strategy will lead to evaluation performance degradation."
                # # +1 to avoid length == self.min_length.
                # idx = random.choice(list(range(0, length - self.min_length + 1, 1)))
                # sample_length = length - idx
                # Resample
                item = random.choice(list(range(self.real_len())))
                name, text, full_motion1, full_motion2, syn_scene, idx, sample_length, length = self.sample_data(item)
                num_times = 0

            if num_times > 300:
                # Resample
                item = random.choice(list(range(self.real_len())))
                name, text, full_motion1, full_motion2, syn_scene, idx, sample_length, length = self.sample_data(item)
                num_times = 0

        # Get feet markers height for ground contact loss.
        if name[0] != 'G':
            smpl_1 = self.smplh_model(return_vertices=True, betas=torch.FloatTensor(full_motion1['betas']).repeat(1, 1))
            smpl_2 = self.smplh_model(return_vertices=True, betas=torch.FloatTensor(full_motion2['betas']).repeat(1, 1))
        else:
            func1 = self.smplx_model_male if full_motion1['gender'] == 'male' else self.smplx_model_female
            func2 = self.smplx_model_male if full_motion2['gender'] == 'male' else self.smplx_model_female
            smpl_1 = func1(return_vertices=True, betas=torch.FloatTensor(full_motion1['betas']).repeat(1, 1))
            smpl_2 = func2(return_vertices=True, betas=torch.FloatTensor(full_motion2['betas']).repeat(1, 1))

        marker_ssm_67 = self.marker_ssm_67 if name[0] != 'G' else self.marker_ssm_67_smplx

        marker_height_1 = \
        (smpl_1.vertices[0, marker_ssm_67][[47, 60, 55, 16, 30, 25]][:, 1] - smpl_1.joints[0, :, 1].min()).detach().numpy()
        marker_height_2 = \
        (smpl_2.vertices[0, marker_ssm_67][[47, 60, 55, 16, 30, 25]][:, 1] - smpl_2.joints[0, :, 1].min()).detach().numpy()

        "Canonicalize the coordinate. Here -1 is for get the feet feature, which needs two near frames."
        full_motion1, full_motion2, syn_scene, motion1_feet, motion2_feet = \
            self.coord_transform.transform_to_canonical(full_motion1, full_motion2, syn_scene, idx, sample_length-1, marker_height_1, marker_height_2,
                                                        is_interx=self.use_interx)

        # motion1 = np.concatenate([full_motion1['marker_ssm2_67'].reshape(-1, 67 * 3), motion1_feet], axis=-1)
        # motion2 = np.concatenate([full_motion2['marker_ssm2_67'].reshape(-1, 67 * 3), motion2_feet], axis=-1)
        # motion1 = full_motion1['marker_ssm2_67'].reshape(-1, 67 * 3)
        # motion2 = full_motion2['marker_ssm2_67'].reshape(-1, 67 * 3)

        motion1 = full_motion1['marker_ssm2_67'] - full_motion1['joints'][:, [0]]
        motion2 = full_motion2['marker_ssm2_67'] - full_motion2['joints'][:, [0]]

        motion1 = np.concatenate([motion1, full_motion1['joints'][:, [0]]], axis=1).reshape(-1, 68 * 3)
        motion2 = np.concatenate([motion2, full_motion2['joints'][:, [0]]], axis=1).reshape(-1, 68 * 3)

        motion1_transf_transl = full_motion1['transf_transl']
        motion1_transf_rotmat = full_motion1['transf_rotmat']
        motion2_transf_transl = full_motion2['transf_transl']
        motion2_transf_rotmat = full_motion2['transf_rotmat']

        gt_length = len(motion1_feet)
        if gt_length < self.max_length:
            padding_len = self.max_length - gt_length
            D = motion1_feet.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            motion1_feet = np.concatenate((motion1_feet, padding_zeros), axis=0)
            motion2_feet = np.concatenate((motion2_feet, padding_zeros), axis=0)

        gt_length = len(motion1)
        if gt_length < self.max_length:
            padding_len = self.max_length - gt_length
            D = motion1.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            motion1 = np.concatenate((motion1, padding_zeros), axis=0)
            motion2 = np.concatenate((motion2, padding_zeros), axis=0)

        assert len(motion1) == self.max_length
        assert len(motion2) == self.max_length

        "Ensure the center person (either person 1/2) is at first."
        if "_swap" in name:
            motion1, motion2 = motion2, motion1
            motion1_transf_rotmat, motion2_transf_rotmat = motion2_transf_rotmat, motion1_transf_rotmat
            motion1_transf_transl, motion2_transf_transl = motion2_transf_transl, motion1_transf_transl
            full_motion1, full_motion2 = full_motion2, full_motion1
            motion1_feet, motion2_feet = motion2_feet, motion1_feet
            marker_height_1, marker_height_2 = marker_height_2, marker_height_1

        # This may contradict the SDF input, since SDF is based on the canonicalized person.
        # if np.random.rand() > 0.5:
        #     motion1, motion2 = motion2, motion1
        #     motion1_transf_rotmat, motion2_transf_rotmat = motion2_transf_rotmat, motion1_transf_rotmat
        #     motion1_transf_transl, motion2_transf_transl = motion2_transf_transl, motion1_transf_transl
        #     full_motion1, full_motion2 = full_motion2, full_motion1
        #     motion1_feet, motion2_feet = motion2_feet, motion1_feet
        #     marker_height_1, marker_height_2 = marker_height_2, marker_height_1

        # Note the pelvis is at the coordinate origin.
        floor_height = -full_motion1['transf_transl'][0, 2]

        # Construct sdf points
        x = torch.linspace(-self.sdf_points_extents, self.sdf_points_extents, self.sdf_points_res)
        y = torch.linspace(-self.sdf_points_extents, self.sdf_points_extents, self.sdf_points_res)
        z = torch.linspace(-self.ceiling_height, self.ceiling_height, self.sdf_points_res)

        xv, yv = torch.meshgrid(x, y)
        points_2d_plane = torch.stack([xv, yv, torch.zeros_like(xv)], axis=2)

        x, y, z = torch.meshgrid(x, y, z)
        points_scene_coord = torch.stack([x, y, z], dim=-1)
        points_scene_sdf = torch.ones_like(points_scene_coord[..., :1])

        # Generate implicit random objects.
        floor_idx = int(floor_height // (self.ceiling_height * 2 / self.sdf_points_res) + 64)
        points_scene_sdf = random_object(points_scene_sdf, floor_idx)

        points_2d = points_2d_plane[..., :2].reshape(self.sdf_points_res ** 2, 1, 2)  # [r*r, 1, 2]
        triangles = torch.FloatTensor(np.stack([syn_scene['vertices'][syn_scene['faces'][:, 0], :2],
                                                     syn_scene['vertices'][syn_scene['faces'][:, 1], :2],
                                                     syn_scene['vertices'][syn_scene['faces'][:, 2], :2]],
                                                    axis=-1)).permute(0, 2, 1)[None, ...]

        d1 = self.sign(points_2d, triangles[:, :, 0, :], triangles[:, :, 1, :])
        d2 = self.sign(points_2d, triangles[:, :, 1, :], triangles[:, :, 2, :])
        d3 = self.sign(points_2d, triangles[:, :, 2, :], triangles[:, :, 0, :])

        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

        inside_triangle = ~(has_neg & has_pos)  # [P, F]
        points_inside_mesh = inside_triangle.any(-1).view(self.sdf_points_res, self.sdf_points_res, 1)

        "To expand contactable area. This can partially prevent the false alarm of the penetration when the marker points are between two sdf points."
        points_inside_mesh = F.max_pool2d(points_inside_mesh.permute(2, 0, 1).float(), kernel_size=3, stride=1, padding=1).permute(1, 2, 0).bool()

        # Make sure the motion part being walkable. This strategy should be with the above generate random objects.
        random_height = (points_inside_mesh * (self.ceiling_height * 2)).repeat(1, 1, self.sdf_points_res)
        # According to random_height matrix, points sdf value below random height are set to -1, others 1.
        points_scene_sdf[(points_scene_coord[:, :, :, 2] + self.ceiling_height) <= random_height] = 1

        "The following commented out script is another strategy that generate random height points outside the " \
        "walkable region. But it cannot achieve good result, maybe due to not follow the real-world situation."
        # points_outside_mesh = ~points_inside_mesh
        #
        # random_height = (points_outside_mesh * (torch.rand((self.sdf_points_res, self.sdf_points_res, 1))
        #                                         * self.ceiling_height * 2)).repeat(1, 1, self.sdf_points_res)
        # # According to random_height matrix, points sdf value below random height are set to -1, others 1.
        # points_scene_sdf[(points_scene_coord[:, :, :, 2] + self.ceiling_height) <= random_height] = -1

        "Dismiss the following ceiling height strategy, as there is case (4861) that one person lie while the other stand, " \
        "leading the stand person in a rather high position"
        scale = random.uniform(2.5, 3.5)
        # ceiling_height = -min(smpl_1.joints[0, 0, 1] - smpl_1.vertices[0, :, 1].max(),
        #                    smpl_2.joints[0, 0, 1] - smpl_2.vertices[0, :, 1].max()).detach().cpu().numpy() * scale
        ceiling_height = np.max(np.concatenate([full_motion1['marker_ssm2_67'][..., 2], full_motion2['marker_ssm2_67'][..., 2]], axis=0), axis=(0,1)) * scale

        # points sdf value below floor or above ceiling are set to -1, others 1.
        "To expand contactable area."
        "Note: For some special cases (4944), when the person sit on the ground, 1 or 2 back markers will below the minimum of the joints positions. Ignore now."
        points_scene_sdf[points_scene_coord[..., 2] < floor_height - self.sdf_points_extents * 2 / self.sdf_points_res] = -1
        points_scene_sdf[points_scene_coord[..., 2] > ceiling_height + self.ceiling_height * 2 / self.sdf_points_res] = -1

        sdf_points = points_scene_sdf.permute(3, 0, 1, 2)

        # Visualize
        # import trimesh
        # import pyrender
        #
        # # scene = pyrender.Scene()
        # # view = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
        #
        # nav_mesh = trimesh.Trimesh(vertices=syn_scene['vertices'], faces=syn_scene['faces'], process=False)
        # # scene.add(pyrender.Mesh.from_trimesh(nav_mesh))
        # mesh_smpl = []
        #
        # if not self.use_interx:
        #     vis_model1 = vis_model2 = smplx.create(r'D:\Motion\Envs\smplx\models', model_type='smplh', gender='neutral', ext='pkl',
        #                              num_betas=10, batch_size=gt_length)
        # else:
        #     if full_motion1['gender'] == 'male':
        #         vis_model1 = smplx.create(r'D:\Motion\Envs\smplx\models', model_type='smplx', gender='male', ext='pkl',
        #                                  num_betas=10, batch_size=gt_length)
        #     else:
        #         vis_model1 = smplx.create(r'D:\Motion\Envs\smplx\models', model_type='smplx', gender='female', ext='pkl',
        #                                  num_betas=10, batch_size=gt_length)
        #     if full_motion2['gender'] == 'male':
        #         vis_model2 = smplx.create(r'D:\Motion\Envs\smplx\models', model_type='smplx', gender='male', ext='pkl',
        #                                  num_betas=10, batch_size=gt_length)
        #     else:
        #         vis_model2 = smplx.create(r'D:\Motion\Envs\smplx\models', model_type='smplx', gender='female', ext='pkl',
        #                                  num_betas=10, batch_size=gt_length)
        # smpl_vis = vis_model1(return_vertices=True, betas=torch.FloatTensor(full_motion1['betas']).repeat(gt_length, 1),
        #                       body_pose=torch.FloatTensor(full_motion1['poses'][:, 3:66]),
        #                       global_orient=torch.FloatTensor(full_motion1['poses'][:, :3]),
        #                       transl=torch.FloatTensor(full_motion1['trans']))
        # smpl_vis2 = vis_model2(return_vertices=True, betas=torch.FloatTensor(full_motion2['betas']).repeat(gt_length, 1),
        #                       body_pose=torch.FloatTensor(full_motion2['poses'][:, 3:66]),
        #                       global_orient=torch.FloatTensor(full_motion2['poses'][:, :3]),
        #                       transl=torch.FloatTensor(full_motion2['trans']))
        #
        # for indx, (v1, v2) in enumerate(zip(smpl_vis.vertices, smpl_vis2.vertices)):
        #     if indx % 10 != 0 and indx != len(smpl_vis.vertices) - 1:
        #         continue
        #     sm = trimesh.Trimesh(vertices=v1.detach().numpy(), faces=vis_model1.faces, process=False)
        #     sm.visual.vertex_colors = [0.8, 0.133, 0.553, indx / len(smpl_vis.vertices) * 0.5 + 0.3]
        #     mesh_smpl.append(sm)
        #
        #     sm = trimesh.Trimesh(vertices=v2.detach().numpy(), faces=vis_model2.faces, process=False)
        #     sm.visual.vertex_colors = [0.154, 0.269, 0.5, indx / len(smpl_vis2.vertices) * 0.5 + 0.3]
        #     mesh_smpl.append(sm)
        #
        #     # mot = trimesh.util.concatenate([trimesh.Trimesh(vertices=v1.detach().numpy(), faces=vis_model.faces, process=False),
        #     #                               trimesh.Trimesh(vertices=v2.detach().numpy(), faces=vis_model.faces, process=False)])
        #     # node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(mot))
        #     # view.render_lock.acquire()
        #     # scene.add_node(node)
        #     # view.render_lock.release()
        #
        #     # time.sleep(0.1)
        #     # view.render_lock.acquire()
        #     # scene.remove_node(node)
        #     # view.render_lock.release()
        #
        # # view.render_lock.acquire()
        # # scene.add(pyrender.Mesh.from_trimesh(trimesh.util.concatenate(mesh_smpl)))
        # # view.render_lock.release()
        #
        # # trimesh sdf_points, and render color according to -1 or 1
        # sdf_points_vis = points_scene_coord[::8, ::8, ::8, :3]
        # sdf_points_color_judge = sdf_points.permute(1, 2, 3, 0)[::8, ::8, ::8, -1]
        # sdf_points_color_judge = sdf_points_color_judge.reshape(-1, 1)
        # sdf_points_vis = sdf_points_vis.reshape(-1, 3).detach().numpy()
        #
        # points_sm_inside = []
        # points_sm_outside = []
        # for i in tqdm(range(0, len(sdf_points_vis), 1)):
        #     tfs = np.eye(4)
        #     tfs[:3, 3] = sdf_points_vis[i]
        #
        #     sm = trimesh.creation.uv_sphere(radius=0.01, transform=tfs)
        #     if sdf_points_color_judge[i] == 1:
        #         sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        #         points_sm_outside.append(sm)
        #     elif sdf_points_color_judge[i] == -1:
        #         sm.visual.vertex_colors = [0.1, 0.9, 0.1, 1.0]
        #         points_sm_inside.append(sm)
        #
        #     # joints_pcl = pyrender.Mesh.from_trimesh(sm)
        #     # view.render_lock.acquire()
        #     # scene.add(joints_pcl)
        #     # view.render_lock.release()
        #     # points_sm.append(sm)
        #
        # points_marker_a = []
        # points_marker_b = []
        # for i in tqdm(range(0, len(full_motion1['marker_ssm2_67'][0]), 1)):
        #     tfs = np.eye(4)
        #     tfs[:3, 3] = full_motion1['marker_ssm2_67'][-1][i]
        #     sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
        #     sm.visual.vertex_colors = [0.1, 0.9, 0.1, 1.0]
        #     points_marker_a.append(sm)
        #
        #     tfs = np.eye(4)
        #     tfs[:3, 3] = full_motion2['marker_ssm2_67'][-1][i]
        #     sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
        #     sm.visual.vertex_colors = [0.1, 0.9, 0.1, 1.0]
        #     points_marker_b.append(sm)
        #
        # points_joints = []
        # for i in tqdm(range(0, len(full_motion1['joints'][0]), 1)):
        #     tfs = np.eye(4)
        #     tfs[:3, 3] = full_motion1['joints'][0][i]
        #     sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
        #     sm.visual.vertex_colors = [0.9, 0.0, 0.1, 1.0]
        #     points_joints.append(sm)
        #
        #     tfs = np.eye(4)
        #     tfs[:3, 3] = full_motion2['joints'][0][i]
        #     sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
        #     sm.visual.vertex_colors = [0.9, 0.0, 0.1, 1.0]
        #     points_joints.append(sm)
        #
        # trimesh.util.concatenate(mesh_smpl).export("human_mesh.ply")
        # trimesh.util.concatenate([nav_mesh]).export("nav_mesh.ply")
        # trimesh.util.concatenate(points_sm_inside).export("points_inside.ply")
        # trimesh.util.concatenate(points_sm_outside).export("points_outside.ply")
        # trimesh.util.concatenate(points_marker_a).export("points_marker_a.ply")
        # trimesh.util.concatenate(points_marker_b).export("points_marker_b.ply")
        #
        # trimesh.util.concatenate(mesh_smpl + [nav_mesh] + points_sm_inside + points_sm_outside + points_marker_a + points_marker_b + points_joints).show()
        # pyrender.Viewer(scene, use_raymond_lighting=True)

        "The output motion has included the conditional marker frame."
        return name, text, motion1, motion2, gt_length - self.cond_length, self.cond_length, \
            np.concatenate((motion1_transf_rotmat, motion2_transf_rotmat), axis=-1), \
            np.concatenate((motion1_transf_transl, motion2_transf_transl), axis=-1), \
            np.concatenate((motion1_feet, motion2_feet), axis=-1), \
            np.concatenate((marker_height_1, marker_height_2), axis=-1), sdf_points

    def sample_data(self, idx):
        data = self.data_list[idx]

        name = data["name"]
        motion_id = data["motion_id"]
        text = random.choice(data["texts"]).strip()

        if self.cache:
            full_motion1, full_motion2, syn_scene = self.motion_dict[motion_id]
            # Deep copy.
            full_motion1 = copy.deepcopy(full_motion1)
            full_motion2 = copy.deepcopy(full_motion2)
            syn_scene = copy.deepcopy(syn_scene)
        else:
            file_path1, file_path2, file_syn_scene = self.motion_dict[motion_id]
            full_motion1 = load_motion(file_path1, self.min_length)
            full_motion2 = load_motion(file_path2, self.min_length)
            syn_scene = load_scene(file_syn_scene)

        length = full_motion1['marker_ssm2_67'].shape[0]
        idx = 0
        sample_length = length
        "Here +1 is for get the feet feature, which needs two near frames."
        if length > self.max_length + 1:
            idx = random.choice(list(range(0, length - (self.max_length + 1), 1)))
            sample_length = self.max_length + 1

        return name, text, full_motion1, full_motion2, syn_scene, idx, sample_length, length