import time

import torch
import torch.nn as nn

from HHInter.utils.utils import *
import smplx
import trimesh
from HHInter.utils.intersection import winding_numbers, pcl_pcl_pairwise_distance
import pickle
import os.path as osp

kinematic_chain = [[0, 2, 5, 8, 11],
                 [0, 1, 4, 7, 10],
                 [0, 3, 6, 9, 12, 15],
                 [9, 14, 17, 19, 21],
                 [9, 13, 16, 18, 20]]

def calc_sdf(vertices, sdf_grids):
    batch_size, seq_len, num_vertices, _ = vertices.shape

    # Rescale vertices to [-1, 1]
    scale = sdf_grids[0, :3].view(3, -1).max(1)[0] - sdf_grids[0, :3].view(3, -1).min(1)[0]
    vertices_scale = vertices / scale * 2
    # Due to the random masked SDF
    if scale[0] == 0:
        vertices_scale = vertices

    # -1 here is due to sdf_points is concat of [x, y, z, sdf_value]
    sdf_values = F.grid_sample(sdf_grids[:, -1:],
                                   vertices_scale[..., [2, 1, 0]].view(batch_size, seq_len * num_vertices, 1, 1, 3), #[2,1,0] permute because of grid_sample assumes different dimension order, see below
                                   padding_mode='border',
                                   align_corners=True
                                   # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                                   ).reshape(batch_size, seq_len, num_vertices)

    "Visualize predicted markers' values in the scene sdf."
    # sdf_transform = sdf_grids.permute(0, 2, 3, 4, 1)[:, ::16, ::16, ::16].reshape(-1, 4)
    # points_sm = []
    # for i in range(0, len(sdf_transform)):
    #     tfs = np.eye(4)
    #     tfs[:3, 3] = sdf_transform[i][:3].cpu().numpy()
    #
    #     sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
    #     if sdf_transform[i][-1].cpu().numpy() == 1:
    #         sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    #     elif sdf_transform[i][-1].cpu().numpy() == -1:
    #         sm.visual.vertex_colors = [0.1, 0.9, 0.1, 1.0]
    #
    #     points_sm.append(sm)
    #
    # for ind in range(0, seq_len):
    #     if postive_ones[0, ind].sum() <= 0 or timestep_mask[0] == 0 or mask[0, ind] == 0:
    #         continue
    #     points_marker = []
    #     marker = vertices[0, ind].reshape(-1, 3)
    #     values = sdf_values[0, ind].view(-1).detach().cpu().numpy()
    #     for i in range(0, len(marker)):
    #         tfs = np.eye(4)
    #         tfs[:3, 3] = marker[i].detach().cpu().numpy()
    #
    #         sm = trimesh.creation.uv_sphere(radius=0.06, transform=tfs)
    #         sm.visual.vertex_colors = values[i].clip(0, 1) * np.array([0.9, 0.1, 0.1, 1.0]) - \
    #                                   values[i].clip(-1, 0) * np.array([0.1, 0.9, 0.1, 1.0])
    #
    #         points_marker.append(sm)
    #
    #     trimesh.util.concatenate(points_sm + points_marker).show()
    #     break

    return sdf_values


class InterLoss(nn.Module):
    def __init__(self, recons_loss, nb_markers, is_normalized):
        super(InterLoss, self).__init__()
        self.nb_markers = nb_markers
        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')

        self.weights = {}
        self.weights["RO"] = 0.001
        self.weights["JA"] = 3
        self.weights["DM"] = 3
        self.weights["SD"] = 3

        self.losses = {}

        self.is_normalized = is_normalized
        self.normalizer = MotionNormalizerTorch()

    def seq_masked_mse(self, prediction, target, mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss

    def mix_masked_mse(self, prediction, target, mask, batch_mask, contact_mask=None, dm_mask=None):
        if dm_mask is not None:
            loss = (self.Loss(prediction, target) * dm_mask).sum(dim=-1, keepdim=True)/ (dm_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        else:
            loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
        if contact_mask is not None:
            loss = (loss[..., 0] * contact_mask).sum(dim=-1, keepdim=True) / (contact_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        loss = (loss * mask).sum(dim=(-1, -2, -3)) / (mask.sum(dim=(-1, -2, -3)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    def forward(self, motion_pred, motion_gt, mask, timestep_mask):
        B, T = motion_pred.shape[:2]
        self.losses["simple"] = self.seq_masked_mse(motion_pred, motion_gt, mask)
        target = motion_gt
        prediction = motion_pred

        if self.is_normalized:
            target = self.normalizer.backward(target)
            prediction = self.normalizer.backward(prediction)

        prediction = prediction.reshape(B, T, 2, 68, 3)
        target = target.reshape(B, T, 2, 68, 3)

        prediction = prediction[:, :, :, :67, :] + prediction[:, :, :, 67:, :]
        target = target[:, :, :, :67, :] + target[:, :, :, 67:, :]

        prediction = prediction.reshape(B, T, 2, -1)
        target = target.reshape(B, T, 2, -1)

        self.pred_g_markers = prediction[..., :self.nb_markers * 3].reshape(B, T, -1, self.nb_markers, 3)
        self.tgt_g_markers = target[..., :self.nb_markers * 3].reshape(B, T, -1, self.nb_markers, 3)

        self.mask = mask
        self.timestep_mask = timestep_mask

        self.forward_distance_map(thresh=1)
        self.forward_joint_affinity(thresh=0.1)
        self.forward_self_distance_map()
        self.forward_relatvie_rot()
        self.accum_loss()


    def forward_relatvie_rot(self):
        r_hip, l_hip = 57, 27
        across = self.pred_g_markers[..., r_hip, :] - self.pred_g_markers[..., l_hip, :]
        across = across / (across.norm(dim=-1, keepdim=True) + 1e-12)
        across_gt = self.tgt_g_markers[..., r_hip, :] - self.tgt_g_markers[..., l_hip, :]
        across_gt = across_gt / (across_gt.norm(dim=-1, keepdim=True) + 1e-12)

        z_axis = torch.zeros_like(across)
        z_axis[..., 2] = 1

        forward = torch.cross(z_axis, across, axis=-1)
        forward = forward / (forward.norm(dim=-1, keepdim=True) + 1e-12)
        forward_gt = torch.cross(z_axis, across_gt, axis=-1)
        forward_gt = forward_gt / (forward_gt.norm(dim=-1, keepdim=True) + 1e-12)

        pred_relative_rot = qbetween_raw(forward[..., 0, :], forward[..., 1, :])
        tgt_relative_rot = qbetween_raw(forward_gt[..., 0, :], forward_gt[..., 1, :])

        self.losses["RO"] = self.mix_masked_mse(pred_relative_rot[..., [0, 3]].unsqueeze(-2),
                                                            tgt_relative_rot[..., [0, 3]].unsqueeze(-2),
                                                            self.mask[..., 0, :].unsqueeze(-1), self.timestep_mask) * self.weights["RO"]


    def forward_distance_map(self, thresh):
        pred_g_markers = self.pred_g_markers.reshape(self.mask.shape[:-1] + (-1,))
        tgt_g_markers = self.tgt_g_markers.reshape(self.mask.shape[:-1] + (-1,))

        pred_g_markers1 = pred_g_markers[..., 0:1, :].reshape(-1, self.nb_markers, 3)
        pred_g_markers2 = pred_g_markers[..., 1:2, :].reshape(-1, self.nb_markers, 3)
        tgt_g_markers1 = tgt_g_markers[..., 0:1, :].reshape(-1, self.nb_markers, 3)
        tgt_g_markers2 = tgt_g_markers[..., 1:2, :].reshape(-1, self.nb_markers, 3)

        pred_distance_matrix = torch.cdist(pred_g_markers1.contiguous(), pred_g_markers2).reshape(
            self.mask.shape[:-2] + (1, -1,))
        tgt_distance_matrix = torch.cdist(tgt_g_markers1.contiguous(), tgt_g_markers2).reshape(
            self.mask.shape[:-2] + (1, -1,))

        distance_matrix_mask = (pred_distance_matrix < thresh).float()

        self.losses["DM"] = self.mix_masked_mse(pred_distance_matrix, tgt_distance_matrix,
                                                                self.mask[..., 0:1, :],
                                                                self.timestep_mask, dm_mask=distance_matrix_mask) * self.weights["DM"]

    def forward_self_distance_map(self):
        pred_g_markers = self.pred_g_markers.reshape(self.mask.shape[:-1] + (-1,))
        tgt_g_markers = self.tgt_g_markers.reshape(self.mask.shape[:-1] + (-1,))

        pred_g_markers1 = pred_g_markers[..., 0:1, :].reshape(-1, self.nb_markers, 3)
        pred_g_markers2 = pred_g_markers[..., 1:2, :].reshape(-1, self.nb_markers, 3)
        tgt_g_markers1 = tgt_g_markers[..., 0:1, :].reshape(-1, self.nb_markers, 3)
        tgt_g_markers2 = tgt_g_markers[..., 1:2, :].reshape(-1, self.nb_markers, 3)

        pred_distance_1_matrix = torch.cdist(pred_g_markers1.contiguous(), pred_g_markers1).reshape(
            self.mask.shape[:-2] + (1, -1,))
        pred_distance_2_matrix = torch.cdist(pred_g_markers2.contiguous(), pred_g_markers2).reshape(
            self.mask.shape[:-2] + (1, -1,))
        tgt_distance_1_matrix = torch.cdist(tgt_g_markers1.contiguous(), tgt_g_markers1).reshape(
            self.mask.shape[:-2] + (1, -1,))
        tgt_distance_2_matrix = torch.cdist(tgt_g_markers2.contiguous(), tgt_g_markers2).reshape(
            self.mask.shape[:-2] + (1, -1,))

        self.losses["SD"] = self.mix_masked_mse(pred_distance_1_matrix, tgt_distance_1_matrix,
                                                                self.mask[..., 0:1, :],
                                                                self.timestep_mask) * self.weights["SD"] + \
                            self.mix_masked_mse(pred_distance_2_matrix, tgt_distance_2_matrix,
                                                                self.mask[..., 0:1, :],
                                                                self.timestep_mask) * self.weights["SD"]

    def forward_joint_affinity(self, thresh):
        pred_g_markers = self.pred_g_markers.reshape(self.mask.shape[:-1] + (-1,))
        tgt_g_markers = self.tgt_g_markers.reshape(self.mask.shape[:-1] + (-1,))

        pred_g_markers1 = pred_g_markers[..., 0:1, :].reshape(-1, self.nb_markers, 3)
        pred_g_markers2 = pred_g_markers[..., 1:2, :].reshape(-1, self.nb_markers, 3)
        tgt_g_markers1 = tgt_g_markers[..., 0:1, :].reshape(-1, self.nb_markers, 3)
        tgt_g_markers2 = tgt_g_markers[..., 1:2, :].reshape(-1, self.nb_markers, 3)

        pred_distance_matrix = torch.cdist(pred_g_markers1.contiguous(), pred_g_markers2).reshape(
            self.mask.shape[:-2] + (1, -1,))
        tgt_distance_matrix = torch.cdist(tgt_g_markers1.contiguous(), tgt_g_markers2).reshape(
            self.mask.shape[:-2] + (1, -1,))

        distance_matrix_mask = (tgt_distance_matrix < thresh).float()

        self.losses["JA"] = self.mix_masked_mse(pred_distance_matrix, tgt_distance_matrix,
                                                                self.mask[..., 0:1, :],
                                                                self.timestep_mask, dm_mask=distance_matrix_mask) * self.weights["JA"]

    def accum_loss(self):
        loss = 0
        for term in self.losses.keys():
            loss += self.losses[term]
        self.losses["total"] = loss
        return self.losses



class GeometricLoss(nn.Module):
    def __init__(self, recons_loss, nb_markers, name):
        super(GeometricLoss, self).__init__()
        self.name = name
        self.nb_markers = nb_markers
        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')

        self.fids = [47, 60, 55, 16, 30, 25]

        self.weights = {}
        self.weights["VEL"] = 30
        self.weights["FC"] = 30

        self.losses = {}

    def mix_masked_mse(self, prediction, target, mask, batch_mask, contact_mask=None, dm_mask=None):
        if dm_mask is not None:
            loss = (self.Loss(prediction, target) * dm_mask).sum(dim=-1, keepdim=True)/ (dm_mask.sum(dim=-1, keepdim=True) + 1.e-7)  # [b,t,p,4,1]
        else:
            loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
        if contact_mask is not None:
            loss = (loss[..., 0] * contact_mask).sum(dim=-1, keepdim=True) / (contact_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        loss = (loss * mask).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    def forward(self, motion_pred, motion_gt, mask, timestep_mask, motion_R, motion_T, motion_feet, feet_height_thresh, is_eval=False):
        B, T = motion_pred.shape[:2]
        target = motion_gt
        prediction = motion_pred

        self.pred_g_markers = prediction[..., :self.nb_markers * 3].reshape(B, T, self.nb_markers, 3)
        self.tgt_g_markers = target[..., :self.nb_markers * 3].reshape(B, T, self.nb_markers, 3)
        self.mask = mask
        self.timestep_mask = timestep_mask

        if not is_eval:
            self.forward_vel()
            self.forward_contact(motion_R, motion_T, motion_feet, feet_height_thresh)
            self.accum_loss()
        else:
            self.forward_contact_eval(motion_R, motion_T, feet_height_thresh)

    def forward_vel(self):
        pred_vel = self.pred_g_markers[:, 1:] - self.pred_g_markers[:, :-1]
        tgt_vel = self.tgt_g_markers[:, 1:] - self.tgt_g_markers[:, :-1]

        pred_vel = pred_vel.reshape(pred_vel.shape[:-2] + (-1,))
        tgt_vel = tgt_vel.reshape(tgt_vel.shape[:-2] + (-1,))

        self.losses["VEL_"+self.name] = self.mix_masked_mse(pred_vel, tgt_vel, self.mask[:, 1:], self.timestep_mask) * self.weights["VEL"]

    def forward_contact(self, motion_R, motion_T, motion_feet, feet_height_thresh):
        feet_vel = self.pred_g_markers[:, 1:, self.fids, :] - self.pred_g_markers[:, :-1, self.fids,:]
        feet_h = torch.einsum('bij,btpj->btpi', motion_R, self.pred_g_markers) + motion_T[:, None, :, :]
        feet_h = feet_h[:, :-1, self.fids, 2].unsqueeze(-1)
        contact = motion_feet[:, 1:-1]  # Filter out the first conditional frame.

        # contact = self.foot_detect(feet_vel, feet_h, 0.001, feet_height_thresh)

        # Use 1: but not -1: is to consider the padded frame.
        self.losses["FC_vel_"+self.name] = self.mix_masked_mse(feet_vel, torch.zeros_like(feet_vel), self.mask[:, 1:],
                                                          self.timestep_mask,
                                                          contact) * self.weights["FC"]
        self.losses["FC_hgt_" + self.name] = self.mix_masked_mse(feet_h, torch.ones_like(feet_h) * feet_height_thresh.unsqueeze(1).unsqueeze(-1),
                                                                 self.mask[:, 1:],
                                                                 self.timestep_mask,
                                                                 contact) * self.weights["FC"]

    def accum_loss(self):
        loss = 0
        for term in self.losses.keys():
            loss += self.losses[term]
        self.losses[self.name] = loss

    def foot_detect(self, feet_vel, feet_h, thres, feet_height_thresh, margin=0.005):
        velfactor, heightfactor = torch.Tensor([thres] * 6).to(feet_vel.device), feet_height_thresh + margin

        feet_x = (feet_vel[..., 0]) ** 2
        feet_y = (feet_vel[..., 1]) ** 2
        feet_z = (feet_vel[..., 2]) ** 2

        contact = (((feet_x + feet_y + feet_z) < velfactor) & (feet_h.squeeze(-1) < heightfactor.unsqueeze(1))).float()
        return contact

    def forward_contact_eval(self, motion_R, motion_T, feet_height_thresh):
        feet_vel = self.pred_g_markers[:, 1:, self.fids, :] - self.pred_g_markers[:, :-1, self.fids,:]
        feet_h = torch.einsum('bij,btpj->btpi', motion_R, self.pred_g_markers) + motion_T[:, None, :, :]
        feet_h = feet_h[:, :-1, self.fids, 2].unsqueeze(-1)

        contact = self.foot_detect(feet_vel, feet_h, 0.001, feet_height_thresh)

        loss = torch.nn.MSELoss(reduction='none')(feet_vel, torch.zeros_like(feet_vel)).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
        loss = (loss[..., 0] * contact).sum(dim=-1, keepdim=True) / (contact.sum(dim=-1, keepdim=True) + 1.e-7)
        loss = (loss * self.mask[:, 1:]).sum(dim=(-1, -2)) / (self.mask[:, 1:].sum(dim=(-1, -2)) + 1.e-7)  # [b]
        loss = (loss * self.timestep_mask).sum(dim=0) / (self.timestep_mask.sum(dim=0) + 1.e-7)
        self.losses["FC_slide_" + self.name] = loss

        # For foot penetration, consider all the penetrated part, thus margin=0, vel_thresh=1e10.
        contact = self.foot_detect(feet_vel, feet_h, 1e10, feet_height_thresh, margin=0)
        loss = torch.nn.L1Loss(reduction='none')(feet_h, torch.ones_like(feet_h) * feet_height_thresh.unsqueeze(1).unsqueeze(-1)).mean(
            dim=-1, keepdim=True)  # [b,t,p,4,1]
        loss = (loss[..., 0] * contact).sum(dim=-1, keepdim=True) / (contact.sum(dim=-1, keepdim=True) + 1.e-7)
        loss = (loss * self.mask[:, 1:]).sum(dim=(-1, -2)) / (self.mask[:, 1:].sum(dim=(-1, -2)) + 1.e-7)  # [b]
        loss = (loss * self.timestep_mask).sum(dim=0) / (self.timestep_mask.sum(dim=0) + 1.e-7)
        self.losses["FC_pene_" + self.name] = loss


class CustomLoss(nn.Module):
    def __init__(self, name):
        super(CustomLoss, self).__init__()
        self.name = name

        self.weights = {}
        self.weights["weight_pene"] = 0.01
        self.weights["weight_contact_feet"] = 1

        self.feet_marker_idx = [47, 60, 55, 16, 30, 25]

        self.losses = {}

    def seq_masked_mse(self, prediction, target, mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss

    def mix_masked_mse(self, loss, mask, batch_mask):
        loss = (loss * mask).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    def forward(self, Y_l, scene_sdf, motion_R, motion_T, mask, timestep_mask):
        Y_l = Y_l.view(Y_l.shape[0], Y_l.shape[1], 67, 3)
        # R0, T0 = motion_R, motion_T

        vertices_l = Y_l

        # Y_w = torch.einsum('bij,btpj->btpi', R0, Y_l) + T0[:, None, :, :]  # [b, t, p, 3]
        nb, nt = Y_l.shape[:2]
        h = 1 / 40  # 1/FPS
        # Y_l_speed = torch.norm(Y_l[:, 2:, :, :2] - Y_l[:, :-2, :, :2], dim=-1) / (2 * h)  # [b, t=9,P=67]

        '''evaluate contact soft'''
        # dist2gp = torch.abs(Y_w[:, :, self.feet_marker_idx, 2].amin(dim=-1, keepdim=True) - 0.02) # feet on floor
        # dist2skat = (Y_l_speed[:, :, self.feet_marker_idx].amin(dim=-1, keepdim=True) - 0.075).clamp(min=0)
        # l_floor = torch.exp(dist2gp)
        # l_skate = torch.exp(dist2skat)

        # penetration reward
        sdf_values = calc_sdf(vertices_l.view(nb, nt, -1, 3), scene_sdf) # [b, t*p]

        """also consider penetration with floor (not directly used counted number, which is unstable)"""
        negative_values = sdf_values * (sdf_values < 0)

        if vertices_l.shape[2] != 67:
            # l_penetration = torch.exp(-((negative_values.sum(dim=-1, keepdim=True) / 512).clip(min=-100)))
            l_penetration = -negative_values.sum(dim=-1, keepdim=True)
        else:
            # l_penetration = torch.exp(-((negative_values.sum(dim=-1, keepdim=True) / 5).clip(min=-10)))
            l_penetration = -negative_values.sum(dim=-1, keepdim=True)

        # l_floor = self.mix_masked_mse(l_floor, mask, timestep_mask)
        # l_skate = self.mix_masked_mse(l_skate, mask[:, 2:], timestep_mask)
        l_penetration = self.mix_masked_mse(l_penetration, mask, timestep_mask)

        loss = {
            # 'r_floor': l_floor * self.weights["weight_contact_feet"],
            # 'r_skate': l_skate * self.weights["weight_contact_feet"],
            self.name + '_r_pene': l_penetration * self.weights["weight_pene"],
        }
        self.losses.update(loss)

        # self.accum_loss()

    def accum_loss(self):
        loss = 0
        for term in self.losses.keys():
            loss += self.losses[term]
        self.losses[self.name] = loss
        
        
class Marker_reproject_loss(nn.Module):
    def __init__(self, recons_loss):
        super(Marker_reproject_loss, self).__init__()
        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')
            
        self.weight = 1

        self.losses = {}

    def seq_masked_mse(self, prediction, target, mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss

    def mix_masked_mse(self, prediction, target, mask, batch_mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
        loss = (loss * mask).sum(dim=(-1, -2, -3)) / (mask.sum(dim=(-1, -2, -3)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    def forward(self, marker_pred, marker_proj, mask, timestep_mask):
        self.losses["reproj"] = self.mix_masked_mse(marker_pred, marker_proj, mask, timestep_mask) * self.weight


class GeneralContactLoss(nn.Module):
    def __init__(
            self,
            region_aggregation_type: str = 'sum',
            squared_dist: bool = False,
            model_type: str = 'smplx',
            body_model_utils_folder: str = 'body_model_utils',
            **kwargs
    ):
        super().__init__()
        """
        Compute intersection and contact between two meshes and resolves.
        """

        self.region_aggregation_type = region_aggregation_type
        self.squared = squared_dist

        self.criterion = self.init_loss()
        self.weight = 1

        # create extra vertex and faces to close back of the mouth to maske
        # the smplx mesh watertight.
        self.model_type = model_type
        faces = torch.load(
            osp.join(body_model_utils_folder, f'{model_type}_faces.pt')
        )

        if self.model_type == 'smplx':
            max_face_id = faces.max().item() + 1
            inner_mouth_verts = pickle.load(open(
                f'{body_model_utils_folder}/smplx_inner_mouth_bounds.pkl', 'rb')
            )
            vert_ids_wt = torch.tensor(inner_mouth_verts[::-1])  # invert order
            self.register_buffer('vert_ids_wt', vert_ids_wt)
            faces_mouth_closed = []  # faces that close the back of the mouth
            for i in range(len(vert_ids_wt) - 1):
                faces_mouth_closed.append([vert_ids_wt[i], vert_ids_wt[i + 1], max_face_id])
            faces_mouth_closed = torch.tensor(np.array(faces_mouth_closed).astype(np.int64), dtype=torch.long,
                                              device=faces.device)
            faces = torch.cat((faces, faces_mouth_closed), 0)

        self.register_buffer('faces', faces)

        # low resolution mesh
        inner_mouth_verts_path = f'{body_model_utils_folder}/lowres_{model_type}.pkl'
        self.low_res_mesh = pickle.load(open(inner_mouth_verts_path, 'rb'))

        self.losses = {}

    def triangles(self, vertices):
        # get triangles (close mouth for smplx)

        if self.model_type == 'smplx':
            mouth_vert = torch.mean(vertices[:, self.vert_ids_wt, :], 1,
                                    keepdim=True)
            vertices = torch.cat((vertices, mouth_vert), 1)

        triangles = vertices[:, self.faces, :]

        return triangles

    def close_mouth(self, v):
        mv = torch.mean(v[:, self.vert_ids_wt, :], 1, keepdim=True)
        v = torch.cat((v, mv), 1)
        return v

    def to_lowres(self, v, n=100):
        lrm = self.low_res_mesh[n]
        v = self.close_mouth(v)

        # trimesh.Trimesh(vertices=v[0].detach().cpu().numpy()[self.low_res_mesh[100]['smplx_vid']],
        #                 faces=self.low_res_mesh[100]['faces']).show()
        # trimesh.Trimesh(vertices=v[0].detach().cpu().numpy()[self.low_res_mesh[500]['smplx_vid']],
        #                 faces=self.low_res_mesh[500]['faces']).show()
        # trimesh.Trimesh(vertices=v[0].detach().cpu().numpy()[self.low_res_mesh[1000]['smplx_vid']],
        #                 faces=self.low_res_mesh[1000]['faces']).show()

        v = v[:, lrm['smplx_vid'], :]
        t = v[:, lrm['faces'].astype(np.int32), :]
        return v, t

    def init_loss(self):
        def loss_func(v1, v2, mask, timestep_mask, factor=1000, wn_batch=True, return_elemtent=False):
            """
            Compute loss between region r1 on meshes v1 and
            region r2 on mesh v2.
            """
            B, T = v1.shape[:2]

            v1 = v1.view((-1,) + v1.shape[2:])
            v2 = v2.view((-1,) + v2.shape[2:])

            nn = 500
            mask = mask.view((-1,)+mask.shape[2:])

            loss = torch.tensor(0.0, device=v1.device)

            if wn_batch:
                # close mouth for self-intersection test
                v1l, t1l = self.to_lowres(v1, nn)
                v2l, t2l = self.to_lowres(v2, nn)

                # compute intersection between v1 and v2, just to get the interier idx, thus no grad.
                with torch.no_grad():
                    exterior = torch.zeros((v1l.shape[0], v1l.shape[1]), device=v1l.device,
                                           dtype=torch.bool)
                    # Split to avoid OOM.
                    half = v1l.shape[1] // 2
                    exterior[:, :half] = winding_numbers(v1l[:, :half, :],
                                                         t2l).ge(0.99)
                    exterior[:, half:] = winding_numbers(v1l[:, half:, :],
                                                         t2l).ge(0.99)
                    interior_v1 = exterior
                    # interior_v1_ori = winding_numbers(v1l, t2l).ge(0.99)

                    exterior = torch.zeros((v2l.shape[0], v2l.shape[1]), device=t1l.device,
                                           dtype=torch.bool)
                    exterior[:, :half] = winding_numbers(v2l[:, :half, :],
                                                         t1l).ge(0.99)
                    exterior[:, half:] = winding_numbers(v2l[:, half:, :],
                                                         t1l).ge(0.99)
                    interior_v2 = exterior
                    # interior_v2_ori = winding_numbers(v2l, t1l).ge(0.99)

            # visu results
            # if True:
            #     for bidx in range(v1l.shape[0]):
            #         crit_v1, crit_v2 = torch.any(interior_v1[bidx]), torch.any(interior_v2[bidx])
            #         if crit_v1 and crit_v2:
            #             import trimesh
            #             mesh1 = trimesh.Trimesh(
            #                 vertices=v2l[bidx].detach().cpu().numpy(),
            #                 faces=self.low_res_mesh[nn]['faces'].astype(np.int32), process=False
            #             )
            #             col = 255 * np.ones((len(v2l[bidx]), 4))
            #             inside_idx = torch.where(interior_v2[bidx])[0].detach().cpu().numpy()
            #             col[inside_idx] = [0, 255, 0, 255]
            #             mesh1.visual.vertex_colors = col
            #             mesh1.show()
            #             # _ = mesh.export('outdebug/interior_v2_lowres.obj')
            #
            #             vmc = self.close_mouth(v1)
            #             mesh2 = trimesh.Trimesh(vertices=vmc[bidx].detach().cpu().numpy(), faces=self.faces.cpu().numpy(), process=False)
            #             col = 255 * np.ones((20940, 4))
            #             col[20908:] = [255, 0, 0, 255]
            #             mesh2.visual.face_colors = col
            #             mesh2.show()
            #             # _ = mesh.export('outdebug/interior_v1_highres.obj')
            #
            #             # Other direction
            #             mesh3 = trimesh.Trimesh(
            #                 vertices=v1l[bidx].detach().cpu().numpy(),
            #                 faces=self.low_res_mesh[nn]['faces'].astype(np.int32), process=False
            #             )
            #             col = 255 * np.ones((len(v2l[bidx]), 4))
            #             inside_idx = torch.where(interior_v1[bidx])[0].detach().cpu().numpy()
            #             col[inside_idx] = [0, 255, 0, 255]
            #             mesh3.visual.vertex_colors = col
            #             mesh3.show()
            #             # _ = mesh.export('outdebug/interior_v1_lowres.obj')
            #
            #             vmc = self.close_mouth(v2)
            #             mesh4 = trimesh.Trimesh(vertices=vmc[bidx].detach().cpu().numpy(), faces=self.faces.cpu().numpy(), process=False)
            #             col = 255 * np.ones((20940, 4))
            #             col[20908:] = [255, 0, 0, 255]
            #             mesh4.visual.face_colors = col
            #             mesh4.show()
            #
            #             trimesh.Scene([mesh1, mesh3]).show()
            #             # _ = mesh.export('outdebug/interior_v2_highres.obj')

            # batch_losses = []
            # for bidx in range(v1l.shape[0]):
            #     if timestep_mask[bidx // T] == 0 or mask[bidx, 0, 0] == 0:
            #         continue
            #
            #     curr_interior_v1 = interior_v1[bidx]
            #     curr_interior_v2 = interior_v2[bidx]
            #     crit_v1, crit_v2 = torch.any(interior_v1[bidx]), torch.any(interior_v2[bidx])
            #
            #     if crit_v1 and crit_v2:
            #         # find vertices that are close to each other between v1 and v2
            #         # squared_dist = pcl_pcl_pairwise_distance(
            #         #    v1[:,interior_v1[bidx],:], v2[:, interior_v2[bidx], :], squared=self.squared
            #         # )
            #         squared_dist_v1v2 = pcl_pcl_pairwise_distance(
            #             v1l[[[bidx]], curr_interior_v1, :], v2l[[bidx]], squared=self.squared)
            #         squared_dist_v2v1 = pcl_pcl_pairwise_distance(
            #             v2l[[[bidx]], curr_interior_v2, :], v1l[[bidx]], squared=self.squared)
            #
            #         v1_to_v2 = (squared_dist_v1v2[0].min(1)[0] * factor) ** 2
            #         # v1_to_v2 = 10.0 * (torch.tanh(v1_to_v2 / 10.0)**2)
            #
            #         v2_to_v1 = (squared_dist_v2v1[0].min(1)[0] * factor) ** 2
            #         # v2_to_v1 = 10.0 * (torch.tanh(v2_to_v1 / 10.0)**2)
            #
            #         batch_losses.append(v1_to_v2.sum())
            #         batch_losses.append(v2_to_v1.sum())
            #
            # if len(batch_losses) > 0:
            #     loss = sum(batch_losses) / len(batch_losses)

            # Convert the above for loop to matrix operations to speed up.
            is_crit = torch.logical_and(torch.any(interior_v1, dim=-1), torch.any(interior_v2, dim=-1))
            squared_dist_v1v2 = pcl_pcl_pairwise_distance(
                v1l, v2l, squared=self.squared) * interior_v1.unsqueeze(-1)
            squared_dist_v2v1 = pcl_pcl_pairwise_distance(
                v2l, v1l, squared=self.squared) * interior_v2.unsqueeze(-1)
            v1_to_v2 = (squared_dist_v1v2.min(2)[0] * factor) ** 2
            v2_to_v1 = (squared_dist_v2v1.min(2)[0] * factor) ** 2
            v1_to_v2 = v1_to_v2 * is_crit.unsqueeze(-1)
            v2_to_v1 = v2_to_v1 * is_crit.unsqueeze(-1)

            v1_to_v2 = v1_to_v2 * mask[:, 0]
            v2_to_v1 = v2_to_v1 * mask[:, 0]

            v1_to_v2 = v1_to_v2.view(B, T, -1) * timestep_mask[:, None, None]
            v2_to_v1 = v2_to_v1.view(B, T, -1) * timestep_mask[:, None, None]

            loss_out = v1_to_v2.view(B * T, -1) + v2_to_v1.view(B * T, -1)
            loss = loss_out.sum() / (((mask[:, 0, 0] * is_crit).view(B, T, -1) * timestep_mask[:, None, None]).sum() + 1.e-7)

            if return_elemtent:
                return ((mask[:, 0, 0] * is_crit).view(B, T, -1) * timestep_mask[:, None, None]).sum(), loss_out
            return loss

        return loss_func

    def forward(self, mask, timestep_mask, **args):
        self.losses['humanpenetration'] = self.criterion(mask=mask, timestep_mask=timestep_mask, **args) * self.weight