import pickle
from HHInter.utils.plot_script import *
from HHInter.utils import paramUtil
import torch
import tqdm

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image
from matplotlib.animation import FFMpegFileWriter
from human_body_prior.body_model.body_model import BodyModel
import numpy as np
from natsort import ns, natsorted

from pytorch3d import transforms
from HHInter.common.quaternion import *
import pyrender
import time
import os, sys


def convert(data):
    # Note that the data here is joint positions, not axis-angle.
    last_dim = data.shape[-1] // 3
    for i in range(last_dim):
        data[..., 1 + i * 3] = -data[..., 1 + i * 3]

    tmp = data[..., list(1 + i * 3 for i in range(last_dim))]
    data[..., list(1 + i * 3 for i in range(last_dim))] = data[..., list(2 + i * 3 for i in range(last_dim))]
    data[..., list(2 + i * 3 for i in range(last_dim))] = tmp
    return data


def inv_convert(data, mode='angle'):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data).float()
    else:
        data = data.float()
    rotate_matrix = torch.tensor([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    if mode == 'angle':
        data = transforms.axis_angle_to_matrix(data)
        # convert yup tp zup.
        data = torch.einsum('ij,bjk->bik', rotate_matrix, data)
        # convert to axis angle
        data = transforms.matrix_to_axis_angle(data)
    elif mode == 'trans':
        data = torch.einsum('ij,bj->bi', rotate_matrix, data)
    else:
        raise ValueError
    return data.numpy()


def deep_copy_npz(original_file_path):
    # Load original .npz file
    with np.load(original_file_path, allow_pickle=True) as original_data:
        # Create a dictionary to store copied data
        copied_data = {}
        for key in original_data.keys():
            # Deep copy each array
            copied_data[key] = np.copy(original_data[key])
    return copied_data

def traverse_imgs(writer, imgs):
    for img in imgs:
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.imshow(img)
        writer.grab_frame()
        plt.pause(0.001)
        plt.clf()


def vis_body_pose_beta(save_path):
    out = FFMpegFileWriter(fps=30.)
    imgs = []
    for fId in tqdm.tqdm(range(body_pose_beta.v.shape[0])):
        body_mesh = trimesh.Trimesh(vertices=body_pose_beta.v[fId], faces=bm.f,
                                    vertex_colors=np.tile(colors['grey'], (6890, 1)))
        body_mesh2 = trimesh.Trimesh(vertices=body_pose_beta2.v[fId], faces=bm.f,
                                     vertex_colors=np.tile(colors['grey'], (6890, 1)))
        if isinstance(mv.viewer, pyrender.OffscreenRenderer):
            mv.set_static_meshes([body_mesh])
            body_image = mv.render(render_wireframe=False)
            imgs.append(body_image)
        else:
            mv.viewer.render_lock.acquire()
            mv.set_static_meshes([body_mesh])
            mv.set_dynamic_meshes([body_mesh2])

            "Visualize the joints positions."
            if visualization_joint:
                node = []
                for id, _ in enumerate(body_pose_beta2.Jtr[fId]):
                    if id < 22:
                        "raw data person1 Jtr"
                        sm = trimesh.creation.uv_sphere(radius=0.03)
                        sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                        tfs = np.tile(np.eye(4), (1, 1, 1))
                        tfs[:, :3, 3] = raw_Jtr[fId][id].numpy()
                        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                        start_node = pyrender.Node(mesh=m, name='start')
                        node.append(start_node)
                        mv.scene.add_node(start_node)

                        "raw data person2 Jtr"
                        sm = trimesh.creation.uv_sphere(radius=0.03)
                        sm.visual.vertex_colors = [0.0, 0.0, 1.0]
                        tfs = np.tile(np.eye(4), (1, 1, 1))
                        tfs[:, :3, 3] = raw_Jtr2[fId][id].numpy()
                        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                        start_node = pyrender.Node(mesh=m, name='start')
                        node.append(start_node)
                        mv.scene.add_node(start_node)

                        "processed data person1 Jtr"
                        sm = trimesh.creation.uv_sphere(radius=0.03)
                        sm.visual.vertex_colors = [0.0, 1.0, 1.0]
                        tfs = np.tile(np.eye(4), (1, 1, 1))
                        tfs[:, :3, 3] = root_pos_init[fId][id]
                        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                        start_node = pyrender.Node(mesh=m, name='start')
                        node.append(start_node)
                        mv.scene.add_node(start_node)

                        "processed data person2 Jtr"
                        sm = trimesh.creation.uv_sphere(radius=0.03)
                        sm.visual.vertex_colors = [0.0, 1.0, 0.0]
                        tfs = np.tile(np.eye(4), (1, 1, 1))
                        tfs[:, :3, 3] = root_pos_init1[fId][id]
                        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                        start_node = pyrender.Node(mesh=m, name='start')
                        node.append(start_node)
                        mv.scene.add_node(start_node)

                        "joint loc to global rot and trans person1 Jtr after converted"
                        sm = trimesh.creation.uv_sphere(radius=0.03)
                        sm.visual.vertex_colors = [.0, .0, .0]
                        tfs = np.tile(np.eye(4), (1, 1, 1))
                        tfs[:, :3, 3] = body_pose_beta.Jtr[fId][id].numpy()
                        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                        start_node = pyrender.Node(mesh=m, name='start')
                        node.append(start_node)
                        mv.scene.add_node(start_node)

                        "joint loc to global rot and trans person2 Jtr after converted"
                        sm = trimesh.creation.uv_sphere(radius=0.03)
                        sm.visual.vertex_colors = [1.0, 1.0, 0.0]
                        tfs = np.tile(np.eye(4), (1, 1, 1))
                        tfs[:, :3, 3] = body_pose_beta2.Jtr[fId][id].numpy()
                        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                        start_node = pyrender.Node(mesh=m, name='start')
                        node.append(start_node)
                        mv.scene.add_node(start_node)

            mv.viewer.render_lock.release()

        # if fId == 0:
        #     input("press to continue")
        if visualization_joint:
            mv.viewer.render_lock.acquire()
            for i in node:
                mv.scene.remove_node(i)
            mv.viewer.render_lock.release()
        # time.sleep(1 / 60)
    if isinstance(mv.viewer, pyrender.OffscreenRenderer):
        figure = plt.figure(figsize=(12, 8))
        plt.ion()
        plt.tight_layout()
        with out.saving(figure, save_path, dpi=100):
            traverse_imgs(out, imgs)
    elif is_record:
        mv.viewer.save_gif(f"{save_path}.gif")


def bilinear_sample_temporal(data, new_time_steps):
    """
    Bilinearly samples temporal data to obtain new time steps.

    Parameters:
    - data: NumPy array with shape (N, D) where N is the original number of time steps, and D is the dimensionality.
    - new_time_steps: Number of desired time steps (M) after bilinear sampling.

    Returns:
    - NumPy array with shape (M, D) representing bilinearly sampled data.
    """

    # Original number of time steps and dimensionality
    N, D = data.shape

    # Calculate the scale factor for bilinear sampling
    scale_factor = (N - 1) / (new_time_steps - 1)

    # Initialize the result array
    result = np.zeros((new_time_steps, D))

    # Bilinear sampling loop
    for i in range(new_time_steps):
        x = i * scale_factor
        x_floor = int(x)
        x_ceil = min(N - 1, x_floor + 1)
        t = x - x_floor

        # Perform linear interpolation along the time dimension
        value_floor = data[x_floor]
        value_ceil = data[x_ceil]
        interpolated_value = (1 - t) * value_floor + t * value_ceil

        result[i] = interpolated_value

    return result


# https://github1s.com/mkocabas/VIBE/blob/HEAD/lib/utils/geometry.py
# TODO: copmpare with rotmat_spin, EGO2EGO transforms, quaternion.py transforms, and pytorch3d transforms
def rot6d_to_rotmat(x):
    x = x.contiguous().view(-1, 3, 2)

    # Normalize the first vector
    b1 = torch.nn.functional.normalize(x[:, :, 0], dim=1, eps=1e-6)

    dot_prod = torch.sum(b1 * x[:, :, 1], dim=1, keepdim=True)
    # Compute the second vector by finding the orthogonal complement to it
    b2 = torch.nn.functional.normalize(x[:, :, 1] - dot_prod * b1, dim=-1, eps=1e-6)

    # Finish building the basis by taking the cross product
    b3 = torch.cross(b1, b2, dim=1)
    rot_mats = torch.stack([b1, b2, b3], dim=-1)

    return rot_mats


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
    imw, imh = 800, 800
    visualization_joint = False
    is_record = False
    visualize_custom = True
    visualize_interx = False
    display_compared_physical = True
    mv = MeshViewer(width=imw, height=imh, use_offscreen=False, record=is_record)

    "If use SMPLX, there will be obvious distortion."
    bm_fname = r'D:\Motion\Dataset\smplh\neutral/model.npz' if not visualize_interx else r'D:\Motion\Envs\smplx\models\smplx\SMPLX_NEUTRAL.npz'
    bm = BodyModel(bm_fname=bm_fname, num_betas=10)
    bm_male = BodyModel(bm_fname=bm_fname.replace("NEUTRAL", "MALE"), num_betas=10)
    bm_female = BodyModel(bm_fname=bm_fname.replace("NEUTRAL", "FEMALE"), num_betas=10)

    # From raw data:
    with open(r"D:\Motion\Dataset\InterGen/motions/10.pkl", "rb") as f:
        data = pickle.load(f)

    A1_beta = betas = torch.from_numpy(data["person1"]['betas']).view(-1, 10)
    template = bm(betas=A1_beta)
    A1 = template.Jtr[:, :4].numpy()

    A2_beta = betas = torch.from_numpy(data["person2"]['betas']).view(-1, 10)
    template = bm(betas=A2_beta)
    A2 = template.Jtr[:, :4].numpy()

    body_pose_beta = bm(pose_body=torch.from_numpy(data["person1"]['pose_body']),
                        betas=torch.from_numpy(data["person1"]['betas']).view(-1, 10).expand(
                            torch.from_numpy(data["person1"]['pose_body']).shape[0], 10),
                        root_orient=torch.from_numpy(data["person1"]["root_orient"]),
                        trans=torch.from_numpy(data["person1"]["trans"]))
    body_pose_beta2 = bm(pose_body=torch.from_numpy(data["person2"]['pose_body']),
                         betas=torch.from_numpy(data["person2"]['betas']).view(-1, 10).expand(
                             torch.from_numpy(data["person2"]['pose_body']).shape[0], 10),
                         root_orient=torch.from_numpy(data["person2"]["root_orient"]),
                         trans=torch.from_numpy(data["person2"]["trans"]))

    body_pose_beta.v = convert(body_pose_beta.v)
    raw_Jtr = convert(body_pose_beta2.Jtr)

    body_pose_beta2.v = convert(body_pose_beta2.v)
    raw_Jtr2 = convert(body_pose_beta.Jtr)

    # joints = body_pose_beta.Jtr[:, :22]
    # data = joints.view(joints.shape[0], -1).numpy()

    # data = convert(data)

    # data = bilinear_sample_temporal(data, 328)

    # vis_body_pose_beta("out.mp4")

    # ==================================================================
    # From processed data:
    with open(r"D:\Motion\Dataset\InterGen\motions_processed/test.txt", "r") as f:
        test_id = f.readlines()
        for i in range(len(test_id)):
            test_id[i] = test_id[i].strip()

    if not visualize_custom:
        if not visualize_interx:
            for name in natsorted(os.listdir(r"D:\Motion\Dataset\InterGen\motions_processed/person1"), alg=ns.PATH):
                # if int(name[:-4].replace("_swap", "")) != 4861:
                #     continue
                data1 = np.load(f"D:/Motion/Dataset/InterGen/motions_processed/person1/{name}")

                rot_6d = data1[..., 62 * 3:62 * 3 + 21 * 6]
                data1 = data1[..., :22 * 3]

                rot_axis_angle = rot6d_to_axis_angle(rot_6d)

                data1 = convert(data1)

                joints = torch.zeros(data1.shape[0], 52, 3)
                joints[:, :22, :3] = torch.from_numpy(data1).view(-1, 22, 3)

                root_pos_init = data1.reshape(-1, 22, 3)

                "Use SVD to calculate the global R, T between two sets of points."
                B = root_pos_init[:, :4]
                root_orient, root_pos, _ = rigid_transform_3D(np.tile(A1, (B.shape[0], 1, 1)), B)

                "The following way cannot calculate the accurate R, only accuate roll around Z-up."
                "Additional note: We also can't simply calculate R with the quaternion between the two forward vectors. Since " \
                "the rotation is usually consisteed of multiple steps, while the quaternion only consider the rotation around " \
                "the axis that is vertical to the two vectors."
                # r_hip, l_hip = [2,1]
                # across = root_pos_init[:, r_hip] - root_pos_init[:, l_hip]
                # across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
                #
                # # forward (3,), rotate around y-axis
                # forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
                # # forward (3,)
                # forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
                #
                # target = np.array([[0, 0, 1]])
                # root_quat_init = qbetween_np(target, forward_init)
                # # quat_axis_change = qbetween_np(np.array([[0, 0, 1]]), np.array([[0, -1, 0]])).repeat(root_quat_init.shape[0], axis=0)
                #
                # root_orient = transforms.quaternion_to_matrix(torch.from_numpy(root_quat_init))

                root_orient = transforms.matrix_to_axis_angle(torch.from_numpy(root_orient))

                "We don't  use root_pos from SVD as the translation, since there will be offsets from the real one."
                body_pose_beta = bm(pose_body=rot_axis_angle,
                                    trans=torch.from_numpy(root_pos_init[:, 0]) - torch.from_numpy(A1[:, 0]).expand(
                                        rot_axis_angle.shape[0], 3), root_orient=root_orient,
                                    betas=A1_beta.expand(rot_axis_angle.shape[0], 10))

                data1 = np.load(f"D:/Motion/Dataset/InterGen/motions_processed/person2/{name}")

                rot_6d = data1[..., 62 * 3:62 * 3 + 21 * 6]
                data1 = data1[..., :22 * 3]

                rot_axis_angle = rot6d_to_axis_angle(rot_6d)

                data1 = convert(data1)

                joints = torch.zeros(data1.shape[0], 52, 3)
                joints[:, :22, :3] = torch.from_numpy(data1).view(-1, 22, 3)

                root_pos_init1 = data1.reshape(-1, 22, 3)

                B = root_pos_init1[:, :4]
                root_orient, root_pos, _ = rigid_transform_3D(np.tile(A2, (B.shape[0], 1, 1)), B)
                # r_hip, l_hip = [2,1]
                # across = root_pos_init1[:, r_hip] - root_pos_init1[:, l_hip]
                # across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
                #
                # # forward (3,), rotate around y-axis
                # forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
                # # forward (3,)
                # forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
                #
                # target = np.array([[0, 0, 1]])
                # root_quat_init = qbetween_np(target, forward_init)
                # # quat_axis_change = qbetween_np(np.array([[0, 0, 1]]), np.array([[0, -1, 0]])).repeat(root_quat_init.shape[0], axis=0)
                #
                # root_orient = transforms.quaternion_to_matrix(torch.from_numpy(root_quat_init))
                root_orient = transforms.matrix_to_axis_angle(torch.from_numpy(root_orient))

                body_pose_beta2 = bm(pose_body=rot_axis_angle,
                                     trans=torch.from_numpy(root_pos_init1[:, 0]) - torch.from_numpy(A2[:, 0]).expand(
                                         rot_axis_angle.shape[0], 3), root_orient=root_orient,
                                     betas=A2_beta.expand(rot_axis_angle.shape[0], 10))

                vis_body_pose_beta(name.split(".")[0])
        else:
            for name in natsorted(os.listdir(r"D:\Motion\Dataset\Inter-X\motions"), alg=ns.PATH):
                data1 = deep_copy_npz(os.path.join(r"D:\Motion\Dataset\Inter-X\motions", name, "P1.npz"))

                if data1['gender'] == "neutral":
                    func1 = bm
                elif data1['gender'] == "female":
                    func1 = bm_female
                elif data1['gender'] == "male":
                    func1 = bm_male
                else:
                    raise ValueError

                # Inter-X is initially in y-up, here convert it to z-up.
                data1['root_orient'] = inv_convert(data1['root_orient'])
                data1['trans'] = inv_convert(data1['trans'], mode='trans')
                # The SMPL rotation is around pelvis, but pelvis is not the original point.
                delta_T_1 = func1(betas=torch.from_numpy(data1['betas']).repeat(1, 1)).Jtr[
                          :, 0, :].detach().cpu().numpy()
                data1['trans'] = data1['trans'] - delta_T_1 + inv_convert(delta_T_1, mode='trans')


                body_pose_beta = func1(pose_body=torch.from_numpy(data1['pose_body'].reshape(-1, 63)),
                                    trans=torch.from_numpy(data1['trans']),
                                    root_orient=torch.from_numpy(data1['root_orient'].reshape(-1, 3)),
                                    betas=torch.from_numpy(data1['betas']).expand(data1['trans'].shape[0], 10),
                        pose_hand=torch.from_numpy(np.concatenate([data1['pose_lhand'], data1['pose_rhand']], axis=1)).reshape(-1, 90))

                smplx_params_a = np.concatenate(
                    [data1['trans'][:, np.newaxis], data1['root_orient'][:, np.newaxis], data1['pose_body'],
                     data1['pose_lhand'],
                     data1['pose_rhand']], axis=1).reshape(len(data1['trans']), 159)
                np.save("P1_smplx_params.npy", smplx_params_a)

                data2 = deep_copy_npz(os.path.join(r"D:\Motion\Dataset\Inter-X\motions", name, "P2.npz"))

                if data2['gender'] == "neutral":
                    func2 = bm
                elif data2['gender'] == "female":
                    func2 = bm_female
                elif data2['gender'] == "male":
                    func2 = bm_male
                else:
                    raise ValueError

                data2['root_orient'] = inv_convert(data2['root_orient'])
                data2['trans'] = inv_convert(data2['trans'], mode='trans')
                delta_T_2 = func2(betas=torch.from_numpy(data2['betas']).repeat(1, 1)).Jtr[
                            :, 0, :].detach().cpu().numpy()
                data2['trans'] = data2['trans'] - delta_T_2 + inv_convert(delta_T_2, mode='trans')

                body_pose_beta2 = func2(pose_body=torch.from_numpy(data2['pose_body'].reshape(-1, 63)),
                                    trans=torch.from_numpy(data2['trans']),
                                    root_orient=torch.from_numpy(data2['root_orient'].reshape(-1, 3)),
                                    betas=torch.from_numpy(data2['betas']).expand(data2['trans'].shape[0], 10),
                        pose_hand=torch.from_numpy(np.concatenate([data2['pose_lhand'], data2['pose_rhand']], axis=1)).reshape(-1, 90))

                smplx_params_b = np.concatenate(
                    [data2['trans'][:, np.newaxis], data2['root_orient'][:, np.newaxis], data2['pose_body'],
                     data2['pose_lhand'],
                     data2['pose_rhand']], axis=1).reshape(len(data2['trans']), 159)
                np.save("P2_smplx_params.npy", smplx_params_b)

                vis_body_pose_beta(name.split(".")[0])
    elif display_compared_physical:
        fold_name = r"D:\Motion\InterGen\results_physical"
        for name in natsorted(os.listdir(fold_name), alg=ns.PATH):
            with open(os.path.join(fold_name, name), "rb") as f:
                data = pickle.load(f).numpy()

            func1 = bm

            data1 = data[0]

            "We don't  use root_pos from SVD as the translation, since there will be offsets from the real one."
            body_pose_beta = func1(pose_body=torch.from_numpy(data1[:, 6:69]),
                                trans=torch.from_numpy(data1[:, :3]), root_orient=torch.from_numpy(data1[:, 3:6]),
                                betas=torch.from_numpy(data1[:, -10:]))

            data2 = data[1]

            body_pose_beta2 = func1(pose_body=torch.from_numpy(data2[:, 6:69]),
                                   trans=torch.from_numpy(data2[:, :3]), root_orient=torch.from_numpy(data2[:, 3:6]),
                                   betas=torch.from_numpy(data2[:, -10:]))

            vis_body_pose_beta(name.split(".")[0])
    else:
        if not visualize_interx:
            fold_name = "InterGen\motions_customized"
        else:
            fold_name = "Inter-X\motions_customized_fps30"
        for name in natsorted(os.listdir(rf"D:\Motion\Dataset\{fold_name}/person1"), alg=ns.PATH):
            # if int(name[:-4].replace("_swap", "")) != 4312:
            #     continue
            data1 = np.load(rf"D:/Motion/Dataset/{fold_name}/person1/{name}")

            if visualize_interx:
                if data1['gender'] == "neutral":
                    func1 = bm
                elif data1['gender'] == "female":
                    func1 = bm_female
                elif data1['gender'] == "male":
                    func1 = bm_male
                else:
                    raise ValueError
            else:
                func1 = bm

            "We don't  use root_pos from SVD as the translation, since there will be offsets from the real one."
            body_pose_beta = func1(pose_body=torch.from_numpy(data1['poses'][:, 3:]),
                                trans=torch.from_numpy(data1['trans']), root_orient=torch.from_numpy(data1['poses'][:, :3]),
                                betas=torch.from_numpy(data1['betas']).expand(data1['trans'].shape[0], 10))

            # body_pose_beta.v = torch.einsum('ij,bpj->bpi', torch.from_numpy(data1['transf_rotmat']).float(), body_pose_beta.v.float()) + torch.from_numpy(data1['transf_transl']).float()[None, :, :]

            data2 = np.load(rf"D:/Motion/Dataset/{fold_name}/person2/{name}")

            if visualize_interx:
                if data2['gender'] == "neutral":
                    func2 = bm
                elif data2['gender'] == "female":
                    func2 = bm_female
                elif data2['gender'] == "male":
                    func2 = bm_male
                else:
                    raise ValueError
            else:
                func2 = bm

            body_pose_beta2 = func2(pose_body=torch.from_numpy(data2['poses'][:, 3:]),
                                trans=torch.from_numpy(data2['trans']), root_orient=torch.from_numpy(data2['poses'][:, :3]),
                                betas=torch.from_numpy(data2['betas']).expand(data2['trans'].shape[0], 10))

            # body_pose_beta2.v = torch.einsum('ij,bpj->bpi', torch.from_numpy(data2['transf_rotmat']).float(), body_pose_beta2.v.float()) + torch.from_numpy(data2['transf_transl']).float()[None, :, :]

            vis_body_pose_beta(name.split(".")[0])

    # plot_3d_motion("result_path.mp4", paramUtil.t2m_kinematic_chain, [data], title="test", fps=30)
