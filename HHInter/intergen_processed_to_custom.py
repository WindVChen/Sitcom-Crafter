import pickle
from collections import OrderedDict

import tqdm
from human_body_prior.body_model.body_model import BodyModel
import numpy as np
from natsort import ns, natsorted

from pytorch3d import transforms
from HHInter.common.quaternion import *
import os, sys



def axis_angle_to_rot6d(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    rot_matrix = transforms.axis_angle_to_matrix(x.view(-1, 21, 3))
    rot_6d = rot_matrix[..., :, :2].clone().reshape(*rot_matrix.size()[:-2], 6)

    return rot_6d


def rot6d_to_axis_angle(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    rot_6d = x.contiguous().view(-1, 3, 2)
    rot_6d = rot_6d.transpose(1, 2)  # 1 X 3 X 2
    rot_6d = rot_6d.contiguous().view(-1, 6)  # 1 X 6

    rot_matrix = transforms.rotation_6d_to_matrix(rot_6d).transpose(1, 2)
    rot_axis_angle = transforms.matrix_to_axis_angle(rot_matrix).view(-1, 63)

    return rot_axis_angle


"Calculate the global R, T between two sets of points."
def rigid_transform_3D(A_all, B_all):
    assert A_all.shape == B_all.shape
    assert len(A_all.shape) == 3

    R_all = []
    t_all = []
    B2_all = []
    for A, B in zip(A_all, B_all):
        N = A.shape[0]
        mu_A = np.mean(A, axis=0)
        mu_B = np.mean(B, axis=0)

        AA = A - np.tile(mu_A, (N, 1))
        BB = B - np.tile(mu_B, (N, 1))
        H = np.dot(np.transpose(AA), BB)

        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            print("Reflection detected")
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = np.dot(-R, mu_A.T) + mu_B.T

        # Calculate error
        B2 = np.dot(R, A.T) + np.tile(t[:, np.newaxis], (1, N))
        B2 = B2.T
        B2_all.append(B2)
        err = B2 - B
        err = np.multiply(err, err).sum()
        # print("err:", err)

        R_all.append(R)
        t_all.append(t)

    R_all = np.stack(R_all)
    t_all = np.stack(t_all)
    B2_all = np.stack(B2_all)
    return R_all, t_all, B2_all


if __name__ == "__main__":
    "If use SMPLX, there will be obvious distortion."
    bm_fname = r'D:\Motion\Dataset\smplh\neutral/model.npz'
    bm = BodyModel(bm_fname=bm_fname, num_betas=10)

    # ==================================================================
    pkl_info = OrderedDict({})

    save_path = r"D:\Motion\Dataset\InterGen\motions_processed_to_custom"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for name in tqdm.tqdm(natsorted(os.listdir(r"D:\Motion\Dataset\InterGen\motions_processed/person1"), alg=ns.PATH)):
        # From raw data:
        with open(rf"D:\Motion\Dataset\InterGen/motions/{name[:-4]}.pkl", "rb") as f:
            data = pickle.load(f)

        A1_beta = torch.from_numpy(data["person1"]['betas']).view(-1, 10)
        template = bm(betas=A1_beta)
        A1 = template.Jtr[:, :4].numpy()

        A2_beta = torch.from_numpy(data["person2"]['betas']).view(-1, 10)
        template = bm(betas=A2_beta)
        A2 = template.Jtr[:, :4].numpy()

        data1 = np.load(f"D:/Motion/Dataset/InterGen/motions_processed/person1/{name}")

        rot_6d = data1[..., 62 * 3:62 * 3 + 21 * 6]
        data1 = data1[..., :22 * 3]

        rot_axis_angle = rot6d_to_axis_angle(rot_6d)

        joints = torch.zeros(data1.shape[0], 52, 3)
        joints[:, :22, :3] = torch.from_numpy(data1).view(-1, 22, 3)

        root_pos_init = data1.reshape(-1, 22, 3)

        "Use SVD to calculate the global R, T between two sets of points."
        B = root_pos_init[:, :4]
        root_orient, root_pos, _ = rigid_transform_3D(np.tile(A1, (B.shape[0], 1, 1)), B)

        root_orient = transforms.matrix_to_axis_angle(torch.from_numpy(root_orient))

        "We don't  use root_pos from SVD as the translation, since there will be offsets from the real one."
        body_pose_beta = bm(pose_body=rot_axis_angle,
                            trans=torch.from_numpy(root_pos_init[:, 0]) - torch.from_numpy(A1[:, 0]).expand(
                                rot_axis_angle.shape[0], 3), root_orient=root_orient,
                            betas=A1_beta.expand(rot_axis_angle.shape[0], 10))

        pkl_info['person1'] = {'betas': A1_beta.squeeze().numpy(), 'pose_body': rot_axis_angle.numpy(),
                               'root_orient': root_orient.numpy(),
                               'trans': (torch.from_numpy(root_pos_init[:, 0]) - torch.from_numpy(A1[:, 0]).expand(rot_axis_angle.shape[0], 3)).numpy(),
                               'gender': 'neutral'}

        data1 = np.load(f"D:/Motion/Dataset/InterGen/motions_processed/person2/{name}")

        rot_6d = data1[..., 62 * 3:62 * 3 + 21 * 6]
        data1 = data1[..., :22 * 3]

        rot_axis_angle = rot6d_to_axis_angle(rot_6d)

        joints = torch.zeros(data1.shape[0], 52, 3)
        joints[:, :22, :3] = torch.from_numpy(data1).view(-1, 22, 3)

        root_pos_init1 = data1.reshape(-1, 22, 3)

        B = root_pos_init1[:, :4]
        root_orient, root_pos, _ = rigid_transform_3D(np.tile(A2, (B.shape[0], 1, 1)), B)

        root_orient = transforms.matrix_to_axis_angle(torch.from_numpy(root_orient))

        body_pose_beta2 = bm(pose_body=rot_axis_angle,
                             trans=torch.from_numpy(root_pos_init1[:, 0]) - torch.from_numpy(A2[:, 0]).expand(
                                 rot_axis_angle.shape[0], 3), root_orient=root_orient,
                             betas=A2_beta.expand(rot_axis_angle.shape[0], 10))

        pkl_info['person2'] = {'betas': A2_beta.squeeze().numpy(), 'pose_body': rot_axis_angle.numpy(),
                               'root_orient': root_orient.numpy(),
                               'trans': (torch.from_numpy(root_pos_init1[:, 0]) - torch.from_numpy(A2[:, 0]).expand(
                                   rot_axis_angle.shape[0], 3)).numpy(),
                               'gender': 'neutral'}

        pkl_info['mocap_framerate'] = 30
        pkl_info['frames'] = data1.shape[0]

        with open(os.path.join(save_path, f"{name[:-4]}.pkl"), "wb") as f:
            pickle.dump(pkl_info, f)

