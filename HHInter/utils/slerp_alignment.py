import numpy as np
import torch
from HHInter.utils.easy_convert import to_matrix, matrix_to


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)
    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
    res = x / (lgth + eps)
    return res


def slerp_poses(last_pose, new_pose, number_of_frames, pose_rep="matrix"):
    # interpolation
    last_pose_matrix = to_matrix(pose_rep, last_pose)
    new_pose_matrix = to_matrix(pose_rep, new_pose)

    last_pose_quat = matrix_to("quaternion", last_pose_matrix).numpy()
    new_pose_quat = matrix_to("quaternion", new_pose_matrix).numpy()

    # SLERP Local Quats:
    interp_ws = np.linspace(0.0, 1.0, num=number_of_frames+2, dtype=np.float32)
    inter = np.stack([(quat_normalize(quat_slerp(quat_normalize(last_pose_quat),
                                                 quat_normalize(new_pose_quat),
                                                 w))) for w in interp_ws], axis=0)
    inter_matrix = to_matrix("quaternion", torch.from_numpy(inter))

    inter_poserep = matrix_to(pose_rep, inter_matrix)
    return inter_poserep[1:-1]

def slerp_translation(last_transl, new_transl, number_of_frames):
    alpha = torch.linspace(0, 1, number_of_frames+2)
    # 2 more than needed
    inter_trans = torch.einsum("i,...->i...", 1-alpha, last_transl) + torch.einsum("i,...->i...", alpha, new_transl)
    return inter_trans[1:-1]

# poses are in matrix format
def aligining_bodies(last_pose, last_trans, poses, transl, pose_rep="matrix"):
    poses_matrix = to_matrix(pose_rep, poses.clone())
    last_pose_matrix = to_matrix(pose_rep, last_pose.clone())

    global_poses_matrix = poses_matrix[:, 0]
    global_last_pose_matrix = last_pose_matrix[0]

    global_poses_axisangle = matrix_to("axisangle", global_poses_matrix)
    global_last_pose_axis_angle = matrix_to("axisangle", global_last_pose_matrix)

    # Find the cancelation rotation matrix
    # First current pose - last pose
    # already in axis angle?
    rot2d_axisangle = global_poses_axisangle[0].clone()
    rot2d_axisangle[:2] = 0
    rot2d_axisangle[2] -= global_last_pose_axis_angle[2]
    rot2d_matrix = to_matrix("axisangle", rot2d_axisangle)

    # turn with the same amount all the rotations
    turned_global_poses_matrix = torch.einsum("...kj,...kl->...jl", rot2d_matrix, global_poses_matrix)
    turned_global_poses = matrix_to(pose_rep, turned_global_poses_matrix)

    turned_poses = torch.cat((turned_global_poses[:, None], poses[:, 1:]), dim=1)

    # turn the trajectory (not with the gravity axis)
    trajectory = transl[:, :2]
    last_trajectory = last_trans[:2]

    vel_trajectory = torch.diff(trajectory, dim=0)
    vel_trajectory = torch.cat((0 * vel_trajectory[[0], :], vel_trajectory), dim=-2)
    vel_trajectory = torch.einsum("...kj,...lk->...lj", rot2d_matrix[:2, :2], vel_trajectory)
    turned_trajectory = torch.cumsum(vel_trajectory, dim=0)

    # align the trajectory
    aligned_trajectory = turned_trajectory + last_trajectory
    aligned_transl = torch.cat((aligned_trajectory, transl[:, [2]]), dim=1)

    return turned_poses, aligned_transl


def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a
    :param x: quaternion tensor
    :param y: quaternion tensor
    :param a: indicator (between 0 and 1) of completion of the interpolation.
    :return: tensor of interpolation results
    """

    len = np.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = np.zeros_like(x[..., 0]) + a
    amount0 = np.zeros(a.shape)
    amount1 = np.zeros(a.shape)

    linear = (1.0 - len) < 0.01
    omegas = np.arccos(len[~linear])
    sinoms = np.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = np.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = np.sin(a[~linear] * omegas) / sinoms
    res = amount0[..., np.newaxis] * x + amount1[..., np.newaxis] * y

    return res

def quat_normalize(x, eps=1e-8):
    """
    Normalizes a quaternion tensor
    :param x: data tensor
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quaternions tensor
    """
    res = normalize(x, eps=eps)
    return res
