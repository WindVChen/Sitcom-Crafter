"""Information
This script is train the policy, given the pre-trained motion models and the body regressors.
- Two motion models: 1-frame-based and 2-frame-based models
- Two body regressors: male or female
- Intermediate results can be saved for visualization.
"""

from logging import log
import warnings
import numpy as np
import random
import argparse
import os, sys, glob
import torchgeometry as tgm
import copy
import pickle
import pdb
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from tqdm import tqdm
from pathlib import Path
import json

sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorCollision, get_map
from exp_GAMMAPrimitive.utils.environments import BatchGeneratorSceneTrain
from exp_GAMMAPrimitive.utils import config_env

from models.models_GAMMA_primitive import GAMMAPrimitiveComboGenOP
from models.models_policy import GAMMAPolicy
from models.baseops import SMPLXParser
from models.baseops import get_logger
from models.chamfer_distance import chamfer_dists

sys.setrecursionlimit(10000)  # if too small, deepcopy will reach the maximal depth limit.

def log_and_print(logstr):
    logger.info(logstr)
    if args.verbose:
        print(logstr)


def configure_model(cfg, gpu_index, seed):
    cfgall = ConfigCreator(cfg)
    modelcfg = cfgall.modelconfig
    traincfg = cfgall.trainconfig
    predictorcfg = ConfigCreator(modelcfg['predictor_config'])
    regressorcfg = ConfigCreator(modelcfg['regressor_config'])

    testcfg = {}
    testcfg['gpu_index'] = gpu_index
    testcfg['ckpt_dir'] = traincfg['save_dir']
    testcfg['result_dir'] = cfgall.cfg_result_dir
    testcfg['seed'] = seed
    testcfg['log_dir'] = cfgall.cfg_log_dir
    testop = GAMMAPrimitiveComboGenOP(predictorcfg, regressorcfg, testcfg)
    testop.build_model(load_pretrained_model=True)

    return testop


def gen_motion_one_step(motion_model, policy_model,
                        states, bparam_seed, prev_betas, scene, obj_points=None, target_ori=None, local_map=None):
    [pred_markers,
     pred_params,
     act,
     act_log_prob,
     value] = motion_model.generate_ppo(policy_model,
                                        states.permute([1, 0, 2]),
                                        bparam_seed.permute([1, 0, 2]),
                                        obj_points=obj_points,
                                        target_ori=target_ori,
                                        local_map=local_map,
                                        n_gens=-1,
                                        betas=prev_betas,
                                        to_numpy=False
                                        )  # [n_gens, t, d]
    pred_markers = pred_markers.reshape(pred_markers.shape[0],
                                        pred_markers.shape[1], -1, 3)  # [n_gens, t, V, 3]
    return pred_markers.permute([1, 0, 2, 3]), pred_params.permute([1, 0, 2]), act, act_log_prob, value


def canonicalize_static_pose(data):
    smplx_transl = data['transl']
    smplx_glorot = data['global_orient']
    smplx_poses = data['body_pose']
    gender = data['gender']
    smplx_handposes = torch.cuda.FloatTensor(smplx_transl.shape[0], 24).zero_()
    prev_params = torch.cat([smplx_transl, smplx_glorot,
                             smplx_poses, smplx_handposes],
                            dim=-1)  # [t,d]
    prev_params = prev_params.repeat(n_gens_1frame, 1, 1)
    prev_betas = data['betas']
    nb, nt = prev_params.shape[:2]
    ## move frame to the body's pelvis
    R0, T0 = smplxparser_1frame.get_new_coordinate(
        betas=prev_betas,
        gender=gender,
        xb=prev_params[:, 0],
        to_numpy=False)

    ## get the last body param and marker in the new coordinate
    body_param_seed = smplxparser_1frame.update_transl_glorot(R0, T0,
                                                              betas=prev_betas,
                                                              gender=gender,
                                                              xb=prev_params.reshape(nb * nt, -1),
                                                              to_numpy=False
                                                              ).reshape(nb, nt, -1)

    return body_param_seed, prev_betas, gender, R0, T0

def get_feature(Y_l, pel, R0, T0, pt_wpath, scene):
    '''
    --Y_l = [b,t,d] local marker
    --pel = [b,t,d]
    --pt_wpath = [1,d]
    '''
    nb, nt = pel.shape[:2]
    Y_l = Y_l.reshape(nb, nt, -1, 3)
    pt_wpath_l_3d = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1), pt_wpath[None, ...] - T0)

    '''extract path feature = normalized direction + unnormalized height'''
    fea_wpathxy = pt_wpath_l_3d[:, :, :2] - pel[:, :, :2]
    fea_wpathxyz = pt_wpath_l_3d[:, :, :3] - pel[:, :, :3]
    dist_xy = torch.norm(fea_wpathxy, dim=-1, keepdim=True).clip(min=1e-12)
    dist_xyz = torch.norm(fea_wpathxyz, dim=-1, keepdim=True).clip(min=1e-12)
    fea_wpathxy = fea_wpathxy / dist_xy
    fea_wpathz = pt_wpath_l_3d[:, :, -1:] - pel[:, :, -1:]
    fea_wpath = torch.cat([fea_wpathxy, fea_wpathz], dim=-1)

    '''extract marker feature'''
    fea_marker = pt_wpath_l_3d[:, :, None, :] - Y_l
    dist_m_3d = torch.norm(fea_marker, dim=-1, keepdim=True).clip(min=1e-12)
    fea_marker_3d_n = (fea_marker / dist_m_3d).reshape(nb, nt, -1)

    '''extract marker feature with depth'''
    dist_m_2d = torch.norm(fea_marker[:, :, :, :2], dim=-1, keepdim=True).clip(min=1e-12)
    fea_marker_xyn = fea_marker[:, :, :, :2] / dist_m_2d
    fea_marker_h = torch.cat([fea_marker_xyn, fea_marker[:, :, :, -1:]], dim=-1).reshape(nb, nt, -1)

    """local map"""
    # t1 = time.time()
    points_local, points_scene, map = get_map(scene['navmesh'],
                          R0, T0,
                          res=cfg_policy.modelconfig.map_res, extent=cfg_policy.modelconfig.map_extent,
                          return_type='torch')
    local_map = map.float()  # [b, res*res]
    local_map[map == False] = -1  #  reassign 0(non-walkable) to -1, https://stats.stackexchange.com/a/138370
    # local_map = torch.zeros_like(local_map)  # test dump map
    # print(time.time() - t1)

    for idx, feature in enumerate((dist_xy, dist_xyz, fea_wpath, fea_marker_3d_n, fea_marker_h, points_local, local_map)):
        if torch.isnan(feature).any() or torch.isinf(feature).any():
            print('feature ', idx, 'is not valid')
            print(feature)

    return dist_xy, dist_xyz, fea_wpath, fea_marker_3d_n, fea_marker_h, points_local, local_map


def get_rewards(bparams, vertices, joints, Y_l, R0, T0, wpath, scene, points_local, local_map, last_step=False):
    t1 = time.time()
    pel_loc = joints[:, :, 0]
    Y_w = torch.einsum('bij,btpj->btpi', R0, Y_l) + T0[:, None, :, :]  # [b, t, p, 3]
    nb, nt = Y_l.shape[:2]
    h = 1 / 40
    Y_l_speed = torch.norm(Y_l[:, 2:] - Y_l[:, :-2], dim=-1) / (2 * h)  # [b, t=9,P=67]
    # Y_l_speed = Y_l_speed.reshape(nb, -1)
    '''evaluate contact soft'''
    # dist2gp = torch.abs(pel_loc[:,-1,-1]-pel_loc[:,0, -1])  # pelvis height not change, not suitable for sitting

    dist2gp = torch.abs(Y_w[:, :, feet_marker_idx, 2].amin(dim=-1) - 0.02).mean(dim=-1)  # feet on floor
    # dist2skat = Y_l_speed.amin(dim=-1)
    dist2skat = (Y_l_speed[:, :, feet_marker_idx].amin(dim=-1) - 0.075).clamp(min=0).mean(dim=-1)
    r_floor = torch.exp(-dist2gp)
    r_skate = torch.exp(-dist2skat)
    # r_contact_friction = r_skate * r_floor

    """penalize static motion"""
    # r_nonstatic = (Y_l_speed[:, :, feet_marker_idx].amax(dim=(1, 2)) > 0.1).float()
    r_nonstatic = (Y_l_speed[:, :, :].mean(dim=(1, 2)) > 0.1).float()

    """penalize when feet are are too far away from pelvis projection on floor"""
    # feet_markers_xy = Y_l[:, :, feet_marker_idx, :2]
    # box_min = feet_markers_xy.amin(dim=[2])  # [b, t, 2]
    # box_max = feet_markers_xy.amax(dim=[2])
    # extents = box_max - box_min
    # # not too far horizontally and vertically
    # dist2big_stride = (extents[:, :, 0] - 0.4).clamp(min=0).mean(dim=-1) + (extents[:, :, 1] - 1).clamp(min=0).mean(dim=-1)
    # r_stride = torch.exp(-dist2big_stride)

    """penalize too fast motion"""
    # dist2run = (Y_l_speed[:, :, feet_marker_idx].amax(dim=-1) - 2).clamp(min=0).mean(dim=-1)
    # r_not_run =torch.exp(dist2run)

    '''evaluate moving direction'''
    pt_wpath = wpath[-1:]
    target_wpath_l = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1), pt_wpath[None, ...] - T0)[:, :, :3]
    b_dir = pel_loc[:, -1, :3] - pel_loc[:, 0, :3]  # (b,3)
    w_dir = target_wpath_l[:, 0] - pel_loc[:, 0, :3]
    w_dir_n = w_dir / torch.norm(w_dir, dim=-1, keepdim=True).clip(min=1e-12)
    b_dir = b_dir / torch.norm(b_dir,dim=-1, keepdim=True).clip(min=1e-12)
    r_move_toward = (1+torch.einsum('bi,bi->b', w_dir_n, b_dir))/2

    """pose prior, reduce unnatural poses"""
    body_pose = bparams[:, :, 6:69].reshape(nt * nb, -1)
    vp_embedding = vposer.encode(body_pose).loc
    latent_dim = vp_embedding.shape[-1]
    vp_norm = torch.norm(vp_embedding.reshape(nb, nt, -1), dim=-1).mean(dim=1)
    r_vp = torch.exp(-vp_norm / (latent_dim ** 0.5))

    """encourage to face the target"""
    joints_end = joints[:, -1]  # [b,p,3]
    x_axis = joints_end[:, 2, :] - joints_end[:, 1, :]
    x_axis[:, -1] = 0
    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True).clip(min=1e-12)
    z_axis = torch.cuda.FloatTensor([[0, 0, 1]], device=device).repeat(x_axis.shape[0], 1)
    y_axis = torch.cross(z_axis, x_axis)
    b_ori = y_axis[:, :2]  # body forward dir of GAMMA is y axis
    face_target_ori = target_wpath_l[:, 0, :2] - pel_loc[:, -1, :2]
    face_target_ori = face_target_ori / torch.norm(face_target_ori, dim=-1, keepdim=True).clip(min=1e-12)
    r_face_target = (torch.einsum('bi,bi->b', face_target_ori, b_ori) + 1) / 2.0

    """distance to target"""
    dist2targetall = torch.norm(target_wpath_l[:, :, :] - pel_loc[:, :, :], dim=-1, keepdim=False).clip(
        min=1e-12)  # [b, t]
    dist2target = torch.norm(target_wpath_l[:, :, :] - pel_loc[:, :, :], dim=-1, keepdim=False).clip(min=1e-12).amin(
        dim=-1)  # [b,]
    r_target_dist = (1 - ((dist2target - cfg.trainconfig.goal_thresh).clip(min=1e-10)) ** 0.5)
    # r_target_dist = torch.exp(-(dist2target - cfg.trainconfig.goal_thresh).clip(min=0))
    # r_target_dist = dist2targetall[:, 0] - dist2targetall[:, -1]
    r_target = r_target_dist * r_face_target

    """penetration according to map"""
    if cfg.lossconfig.pene_type == 'foot':
        markers_local_xy = Y_l[:, :, feet_marker_idx, :2]  # [b, t, p, 2]
    elif cfg.lossconfig.pene_type == 'body':
        markers_local_xy = Y_l[:, :, :, :2]  # [b, t, p, 2]
    # get body bbox on xy plane
    box_min = markers_local_xy.amin(dim=[1, 2]).reshape(nb, 1, 2)
    box_max = markers_local_xy.amax(dim=[1, 2]).reshape(nb, 1, 2)
    inside_feet_box = ((points_local[:, :, :2] >= box_min).all(-1) & (points_local[:, :, :2] <= box_max).all(-1)).float()
    num_pene = inside_feet_box * (1 - local_map) * 0.5# [b, D]
    num_pene = num_pene.sum(dim=[1]) # [b]
    r_pene = torch.exp(-num_pene * 2)

    """check whether reach the goal within distance and orientation threshold"""
    use_early_stop = cfg.trainconfig.use_early_stop
    terminate = True if use_early_stop and (dist2target < cfg.trainconfig.goal_thresh).any() else False
    r_success = (dist2target < cfg.trainconfig.goal_thresh).float()
    if not last_step and not terminate and cfg.lossconfig.sparse_reward:
        r_target_dist = 0

    reward = r_floor * cfg.lossconfig.weight_floor + \
             r_skate * cfg.lossconfig.weight_skate + \
             r_vp * cfg.lossconfig.weight_vp + \
             r_pene * cfg.lossconfig.weight_pene + \
             r_target_dist * cfg.lossconfig.weight_target_dist + \
             r_face_target * cfg.lossconfig.weight_face_target + \
             r_nonstatic * cfg.lossconfig.weight_nonstatic + \
             r_success * cfg.lossconfig.weight_success

    for idx, r_value in enumerate((r_floor, r_skate, r_vp, r_pene, r_target_dist, r_face_target, r_nonstatic, r_success)):
        if torch.isnan(r_value).any() or torch.isinf(r_value).any():
            print('reward ', idx, 'is not valid')
            print(r_value)

        # reward = reward + r_floor + r_pelvis_vel
    if last_step or terminate or not cfg.lossconfig.sparse_reward:
        wandb.log({
            'r_target_dist': r_target_dist.mean(),
            'r_target': r_target.mean(),
            'r_success': r_success.mean(),
        })
    wandb.log({
        'r_floor': r_floor.mean(),
        'r_skate': r_skate.mean(),
        'r_nonstatic': r_nonstatic.mean(),
        'r_vp': r_vp.mean(),
        'r_move_toward': r_move_toward.mean(),
        'r_face_target': r_face_target.mean(),
        'r_pene': r_pene.mean(),
        'num_pene': num_pene.mean(),
        'reward': reward.mean(),
    })
    return reward, terminate


def gen_tree_roots(data_mp, wpath, scene):
    body_param_seed, prev_betas, gender, R0, T0 = data_mp
    nb, nt = body_param_seed.shape[:2]
    t_his = 1

    """retrieve current motion model"""
    motion_model = genop_1frame_male if gender == 'male' else genop_1frame_female
    """retrieve current states"""

    marker_seed = smplxparser_1frame.get_markers(
        betas=prev_betas,
        gender=gender,
        xb=body_param_seed.reshape(nb * nt, -1),
        to_numpy=False
    ).reshape(nb, nt, -1)

    pelvis_loc = smplxparser_1frame.get_jts(betas=prev_betas,
                                            gender=gender,
                                            xb=body_param_seed.reshape(nb * nt, -1),
                                            to_numpy=False
                                            )[:, 0]  # [b*t, 3]
    pelvis_loc = pelvis_loc.reshape(nb, nt, -1)

    distxy, dist, fea_wpath, fea_marker, fea_marker_h, points_local, local_map = get_feature(marker_seed, pelvis_loc[:, :t_his],
                                                                                     R0, T0, wpath[-1:], scene)
    # if np.any(dist.detach().cpu().numpy() < GOAL_THRESH):
    #     warnings.warn('[warning] the current target is too close to the starting location!')
    #     return None
    states = torch.cat([marker_seed, fea_marker], dim=-1) if 'condi' in cfg_policy.modelconfig['body_repr'] else marker_seed
    # generate markers and regress to body params
    [pred_markers, pred_params,
     act, act_log_prob, value] = gen_motion_one_step(
        motion_model, policy_model,
        states, body_param_seed, prev_betas, scene,
        local_map=local_map if 'map' in cfg_policy.modelconfig['body_repr'] else None,
    )
    nb, nt = pred_params.shape[:2]
    pred_joints = smplxparser_mp.get_jts(betas=prev_betas,
                                         gender=gender,
                                         xb=pred_params.reshape([nb * nt, -1]),
                                         to_numpy=False).reshape([nb, nt, -1, 3])
    pred_pelvis_loc = pred_joints[:, :, 0]
    pred_markers_proj = smplxparser_mp.get_markers(betas=prev_betas,
                                                   gender=gender,
                                                   xb=pred_params.reshape([nb * nt, -1]),
                                                   to_numpy=False).reshape((nb, nt, -1, 3))  # [b, t, p, 3]

    pred_marker_b = REPROJ_FACTOR * pred_markers_proj + (1 - REPROJ_FACTOR) * pred_markers
    pred_vertices = smplxparser_mp.forward_smplx(betas=prev_betas,
                                                 gender=gender,
                                                 xb=pred_params.reshape([nb * nt, -1]),
                                                 output_type='vertices',
                                                 to_numpy=False).reshape([nb, nt, -1, 3])
    traj_rewards, terminate = get_rewards(pred_params, pred_vertices, pred_joints,
                                           pred_marker_b, R0, T0, wpath, scene, points_local, local_map)
    traj_states = states  # [b,t,d]
    traj_local_map = local_map # [b, 2]
    traj_act = act  # [b,d]
    traj_act_logprob = act_log_prob  # [b,]
    traj_value = value  # [b,]
    return [[traj_states, traj_local_map, traj_act, traj_act_logprob, traj_rewards, traj_value],
            [pred_marker_b, pred_params, prev_betas, gender, R0, T0, pred_pelvis_loc, '1-frame']]


def expand_tree(data_mp, wpath, scene, last_step=False):
    prev_markers_b, prev_params, prev_betas, prev_gender, prev_rotmat, prev_transl, prev_pelvis_loc, _ = data_mp

    t1 = time.time()
    # ===================== produce marker seed =====================#
    t_his = 2
    body_param_seed = prev_params[:, -t_his:]  # [b,t,d]
    nb, nt = body_param_seed.shape[:2]
    ## move frame to the second last body's pelvis
    R_, T_ = smplxparser_1frame.get_new_coordinate(
        betas=prev_betas,
        gender=prev_gender,
        xb=body_param_seed[:, 0],
        to_numpy=False)  # [b,3,3][b,1,3]
    T0 = torch.einsum('bij,btj->bti', prev_rotmat, T_) + prev_transl
    R0 = torch.einsum('bij,bjk->bik', prev_rotmat, R_)
    body_param_seed = smplxparser_2frame.update_transl_glorot(
        R_.repeat(t_his, 1, 1),
        T_.repeat(t_his, 1, 1),
        betas=prev_betas,
        gender=prev_gender,
        xb=body_param_seed.reshape(nb*nt, -1),
        to_numpy=False,
        inplace=False).reshape(nb, nt, -1)

    marker_seed = torch.einsum('bij,btpj->btpi', R_.permute(0, 2, 1), prev_markers_b[:, -t_his:] - T_[..., None, :])
    pel_loc_seed = torch.einsum('bij,btj->bti', R_.permute(0, 2, 1), prev_pelvis_loc[:, -t_his:] - T_)
    distxy, dist, fea_wpath, fea_marker, fea_marker_h, points_local, local_map = get_feature(marker_seed, pel_loc_seed, R0, T0,
                                                                                     wpath[-1:], scene)
    dist2target = dist[:, -1, 0]
    # if USE_EARLY_STOP:
    #     if torch.any(dist2target < GOAL_THRESH):
    #         # log_and_print('[INFO] some motion reaches the target. quit rollout and collect rewards')
    #         return dist2target

    # ===================== generate future marker =====================#
    motion_model = genop_2frame_male if prev_gender == 'male' else genop_2frame_female
    marker_seed = marker_seed.reshape(nb, t_his, -1)  # [b,t,d]
    states = torch.cat([marker_seed, fea_marker], dim=-1) if 'condi' in cfg_policy.modelconfig['body_repr'] else marker_seed
    # generate markers and regress to body params
    [pred_markers, pred_params,
     act, act_log_prob, value] = gen_motion_one_step(
        motion_model, policy_model,
        states, body_param_seed, prev_betas, scene,
        local_map=local_map if 'map' in cfg_policy.modelconfig['body_repr'] else None,
    )

    nb, nt = pred_params.shape[:2]
    pred_joints = smplxparser_mp.get_jts(betas=prev_betas,
                                         gender=prev_gender,
                                         xb=pred_params.reshape([nb * nt, -1]),
                                         to_numpy=False).reshape([nb, nt, -1, 3])  # [t*b, 3]
    pred_pelvis_loc = pred_joints[:, :, 0]
    pred_markers_proj = smplxparser_mp.get_markers(betas=prev_betas,
                                                   gender=prev_gender,
                                                   xb=pred_params.reshape([nb * nt, -1]),
                                                   to_numpy=False).reshape((nb, nt, -1, 3))  # [b, t, p, 3]

    pred_marker_b = REPROJ_FACTOR * pred_markers_proj + (1 - REPROJ_FACTOR) * pred_markers
    pred_vertices = smplxparser_mp.forward_smplx(betas=prev_betas,
                                                 gender=prev_gender,
                                                 xb=pred_params.reshape([nb * nt, -1]),
                                                 output_type='vertices',
                                                 to_numpy=False).reshape([nb, nt, -1, 3])
    traj_rewards, terminate = get_rewards(pred_params, pred_vertices, pred_joints, pred_marker_b, R0, T0,
                               wpath, scene, points_local, local_map,
                               last_step=last_step)
    traj_states = states  # [b,t,d]
    traj_local_map = local_map # [b, 2]
    traj_act = act  # [b,d]
    traj_act_logprob = act_log_prob  # [b,]
    traj_value = value  # [b,]

    return [[traj_states, traj_local_map, traj_act, traj_act_logprob, traj_rewards, traj_value],
            dist2target, terminate,
            [pred_marker_b, pred_params, prev_betas, prev_gender, R0, T0, pred_pelvis_loc, '2-frame']]


def discount_rewards(rewards: list, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [rewards[-1]]
    for i in reversed(range(len(rewards) - 1)):
        new_rewards.append(rewards[i] + gamma * new_rewards[-1])
    return new_rewards[::-1]


def calculate_aes(returns: list, values: list):
    aes = []
    for i in range(len(returns)):
        ae = returns[i] - values[i]
        ae = (ae - ae.mean()) / (ae.std() + 1e-10)
        aes.append(ae)
    return aes


def save_rollout_results(wpath, scene, outmps, outfolder):
    # print(outfolder)
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    mp_keys = ['blended_marker', 'smplx_params', 'betas', 'gender', 'transf_rotmat', 'transf_transl', 'pelvis_loc',
               'mp_type']

    for b in range(n_gens_1frame):
        outmps_nodes = {'motion': [], 'wpath': wpath.detach().cpu().numpy(),
                        # 'target_orient': scene['target_orient'].detach().cpu().numpy(),
                        'navmesh_path': scene['navmesh_path'],

                        }
        if 'obj_id' in scene:
            outmps_nodes['obj_id'] = scene['obj_id']
        if 'obj_transform' in scene:
            outmps_nodes['obj_transform'] = scene['obj_transform'],
        if 'scene_path' in scene:
            outmps_nodes['scene_path'] = scene['scene_path']
        for mp in outmps:
            mp_node = {}
            for idx, key in enumerate(mp_keys):
                if key in ['gender', 'mp_type', 'betas']:
                    mp_node[key] = mp[idx] if type(mp[idx]) == str else mp[idx].detach().cpu().numpy()
                elif key in ['smplx_params']:
                    mp_node[key] = mp[idx][b:b + 1].detach().cpu().numpy()
                else:
                    mp_node[key] = mp[idx][b].detach().cpu().numpy()
            outmps_nodes['motion'].append(mp_node)
        with open(outfolder + '/motion_{:02d}.pkl'.format(b), 'wb') as f:
            pickle.dump(outmps_nodes, f)


def rollout(body_s, max_depth=10, epoch=0):
    '''generate motion and collect data for policy update.
    '''
    traj_ppo = {'states': [],
                'local_map': [],
                'actions': [],
                'action_logprob': [],
                'rewards': [],
                'value': []}

    outmps = []
    # specify the start node
    data_mp0 = canonicalize_static_pose(body_s)
    # wpath = torch.cuda.FloatTensor(body_s['wpath'], device=device) #numpy, 2x3
    wpath = body_s['wpath']  # 2x3
    scene = body_s
    stime = time.time()

    """generating tree roots"""
    rootout = gen_tree_roots(data_mp0, wpath, scene)
    if rootout is not None:
        data_traj, data_mp = rootout
    else:
        return None
    outmps.append(data_mp)
    for i, key in enumerate(traj_ppo.keys()):
        traj_ppo[key].append(data_traj[i])

    """expanding tree"""
    dist2target_record = []
    for step in range(1, max_depth):
        # print('[info] current depth={}'.format(depth))
        mp_new = expand_tree(data_mp, wpath, scene, last_step=(step == max_depth - 1))
        if type(mp_new) == list:
            data_traj, dist2target_, terminate, data_mp = mp_new
            dist2target_record.append(dist2target_)
            outmps.append(data_mp)
            for i, key in enumerate(traj_ppo.keys()):
                traj_ppo[key].append(data_traj[i])
            if terminate:
                # print('reach goal, early stop')
                break
        else:
            dist2target_ = mp_new
            dist2target_record.append(dist2target_)
            break

    dist2targetall = torch.stack(dist2target_record, dim=1)

    """give bonus according to the dist2target"""
    dist2targetall[dist2targetall <= GOAL_THRESH] = 0
    # dist2target = torch.amin(dist2targetall,dim=-1)
    dist2target = dist2targetall[:, -1]  # check last frame distance
    # if args.last_only:
    #     traj_ppo['rewards'][-1] += (1 - (dist2target / args.active_range)).clamp(min=0) * args.weight_target_dist
    # else:
    #     for i, rew in enumerate(traj_ppo['rewards']):
    #         traj_ppo['rewards'][i] = rew + (1 - (dist2target / args.active_range) ** 0.5) * args.weight_target_dist

    'compute returns from traj_rewards'
    traj_ppo['returns'] = discount_rewards(traj_ppo['rewards'],
                                           gamma=REWARD_DISCOUNT)  # 1d array
    'a simple advantage function (ae)'
    traj_ppo['gae'] = calculate_aes(traj_ppo['returns'],
                                    traj_ppo['value'])  # 1d array

    """logging"""
    wandb.log({'return': torch.stack(traj_ppo['returns']).mean()})
    # wandb.log({'gae': torch.stack(traj_ppo['gae']).mean()})
    wandb.log({
        'avg_rewards': np.mean([rew.detach().cpu().numpy().mean() for rew in traj_ppo['rewards']]),
        'avg_dist': np.mean(dist2target.detach().cpu().numpy()),
        'worst_rewards': np.min([rew.detach().cpu().numpy().min() for rew in traj_ppo['rewards']]),
        'worst_dist': np.amax(dist2target.detach().cpu().numpy()),
        'best_dist': np.amin(dist2target.detach().cpu().numpy()),
    })
    if epoch % cfg.trainconfig.log_interval == 0:
        depth = len(traj_ppo['states'])
        epstime = time.time() - stime
        info_str = '[epoch {}][INFO, ROLLOUT] steps={:d}/{:d}, success_ratio={:.02f}, avg_rewards={:.02f}, avg_dist={:.02f}, worst_rewards={:.02f}, worst_dist={:.02f}, best_dist={:.02f}, epstime={:.02f}'.format(
            epoch,
            depth, max_depth, np.mean(dist2target.detach().cpu().numpy() < GOAL_THRESH),
            np.mean([rew.detach().cpu().numpy().mean() for rew in traj_ppo['rewards']]),
            np.mean(dist2target.detach().cpu().numpy()),
            np.min([rew.detach().cpu().numpy().min() for rew in traj_ppo['rewards']]),
            np.amax(dist2target.detach().cpu().numpy()),
            np.amin(dist2target.detach().cpu().numpy()),
            epstime
            )
        log_and_print(info_str)

        """visualze rollout results in blender, for debugging"""
        save_rollout_results(wpath, scene, outmps, os.path.join(cfg.cfg_result_dir, 'epoch' + str(epoch)))

    return traj_ppo


def calc_loss_policy_val(traj_states,
                         traj_local_map,
                         traj_z,
                         traj_z_logprob,
                         traj_gae,
                         traj_returns):
    '''forward pass of the policy value model'''
    z_mu, z_logvar, value = policy_model(traj_states.permute([1, 0, 2]), local_map=traj_local_map)
    z_var = torch.exp(z_logvar.clamp(policy_model.min_logvar, policy_model.max_logvar))
    act_distrib_c = Normal(z_mu, z_var ** 0.5)  # batched distribution
    act_distrib = Independent(act_distrib_c, 1)

    '''calculate ppo'''
    traj_z_logprob_new = act_distrib.log_prob(traj_z)  # size=(b=1, )
    policy_ratio = torch.exp(traj_z_logprob_new - traj_z_logprob)
    if torch.isnan(policy_ratio).any() or torch.isinf(policy_ratio).any():
        print('policy ratio error!', traj_z_logprob_new, traj_z_logprob)
    clipped_ratio = policy_ratio.clamp(1 - PPO_CLIP_VAL,
                                       1 + PPO_CLIP_VAL)
    clipped_loss = clipped_ratio * traj_gae
    full_loss = policy_ratio * traj_gae
    loss_ppo = -torch.min(full_loss, clipped_loss).mean()
    if torch.isnan(loss_ppo).any() or torch.isinf(loss_ppo).any():
        print('ppo loss', full_loss, clipped_loss)

    '''KLD like in Motion Primitive VAE'''
    # loss_kld = 0.5 * torch.mean(-1 - z_logvar + z_mu.pow(2) + z_logvar.exp())  # batchmean is mathematically correct. https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    loss_kld = 0.5 * torch.mean(-1 - z_logvar + z_mu.pow(2) + z_var)  # batchmean is mathematically correct. https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    if torch.isnan(loss_kld).any() or torch.isinf(loss_kld).any():
        print('kld loss nan!', z_logvar, z_mu)
    if USE_ROBUST_KLD:
        loss_kld = (torch.sqrt(1 + loss_kld ** 2) - 1)
        if torch.isnan(loss_kld).any():
            print('kld loss nan!', z_logvar, z_mu)
    wandb.log({
        'kl': loss_kld,
    })
    loss_kld = cfg_policy.lossconfig['kld_weight'] * loss_kld
    '''check kld stateus'''
    kld_thresh = (traj_z_logprob - traj_z_logprob_new).mean()

    '''calculate value'''
    loss_val = torch.mean((value - traj_returns) ** 2)

    '''prepare output'''
    loss = loss_ppo + loss_val + loss_kld
    loss_items = np.array([loss_ppo.item(), loss_val.item()])
    return loss, loss_items, kld_thresh


@hydra.main(version_base=None, config_path="../exp_GAMMAPrimitive/cfg", config_name="MPVAEPolicy_babel_collision")
def get_cfg(omega_cfg: DictConfig):
    global cfg
    cfg = omega_cfg
    print(OmegaConf.to_yaml(cfg))


# create dirs for saving, then add to cfg
def create_dirs(cfg):
    with open_dict(cfg):
        # create dirs
        cfg.cfg_exp_dir = os.path.join('results', 'exp_GAMMAPrimitive', cfg.cfg_name, cfg.wandb.name)
        cfg.cfg_result_dir = os.path.join(cfg.cfg_exp_dir, 'results')
        cfg.cfg_ckpt_dir = os.path.join(cfg.cfg_exp_dir, 'checkpoints')
        cfg.cfg_log_dir = os.path.join(cfg.cfg_exp_dir, 'logs')
        os.makedirs(cfg.cfg_result_dir, exist_ok=True)
        os.makedirs(cfg.cfg_ckpt_dir, exist_ok=True)
        os.makedirs(cfg.cfg_log_dir, exist_ok=True)

        # specify missed experiment settings
        cfg['trainconfig']['save_dir'] = cfg.cfg_ckpt_dir
        cfg['trainconfig']['log_dir'] = cfg.cfg_log_dir

if __name__ == '__main__':
    """get cfg"""
    cfg = None
    get_cfg()
    create_dirs(cfg)
    OmegaConf.save(cfg, Path(cfg.cfg_exp_dir, "config.yaml"))
    args = cfg.args
    args.exp_name = cfg.wandb.name
    cfg_policy = cfg

    """init wandb to log metrics, hparams, models and code"""
    wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
               name=cfg.wandb.name, group=cfg.wandb.group,
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
               )
    wandb.run.log_code("../models", name="models")
    wandb.run.log_code(".", name="exp", include_fn=lambda path: path.endswith(".py"))

    """setup"""
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda',
                          index=args.gpu_index) if args.gpu_index >= 0 and torch.cuda.is_available() else torch.device(
        'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(True)

    # cfg_policy = ConfigCreator(args.cfg_policy)
    cfg_1frame_male = cfg_policy.trainconfig['cfg_1frame_male']
    cfg_2frame_male = cfg_policy.trainconfig['cfg_2frame_male']
    cfg_1frame_female = cfg_policy.trainconfig['cfg_1frame_female']
    cfg_2frame_female = cfg_policy.trainconfig['cfg_2frame_female']
    max_depth = cfg_policy.trainconfig['max_depth']
    print('max_depth:', max_depth)
    GOAL_THRESH = cfg_policy.trainconfig['goal_thresh']
    GOAL_SIGMA=cfg_policy.trainconfig['goal_disturb_sigma']
    n_gens_1frame = cfg_policy.trainconfig['n_gens_1frame']
    n_gens_2frame = cfg_policy.trainconfig['n_gens_2frame']
    lr_po = cfg_policy.trainconfig['learning_rate_p']
    lr_val = cfg_policy.trainconfig['learning_rate_v']
    max_train_iter_1f = cfg_policy.trainconfig['max_train_iter_1f']
    max_train_iter_2f = cfg_policy.trainconfig['max_train_iter_2f']
    body_repr = cfg_policy.modelconfig['body_repr']

    num_epochs = cfg_policy.trainconfig['num_epochs']
    num_envs_per_epoch = cfg_policy.trainconfig.get('num_envs_per_epoch', 1)
    saving_per_X_ep = cfg_policy.trainconfig['saving_per_X_ep']
    REWARD_DISCOUNT = cfg_policy.lossconfig['reward_discount']
    GAE_DECAY = cfg_policy.lossconfig['gae_decay']
    PPO_CLIP_VAL = cfg_policy.lossconfig['ppo_clip_val']
    logger = get_logger(cfg_policy.trainconfig['log_dir'])
    POSITIVE_MINING = cfg_policy.trainconfig.get('positive_mining', False)
    KLD_THRESH = cfg_policy.lossconfig.get('kld_thresh', 0.05)
    BATCH_SIZE = cfg_policy.trainconfig.get('batch_size', 1024)
    USE_FACING_REWARD = cfg_policy.lossconfig.get('use_facing_reward', False)
    USE_VPOSER_REWARD = cfg_policy.lossconfig.get('use_vposer_reward', False)
    USE_NORMALIZED_MOVEMENT = cfg_policy.lossconfig.get('use_normalized_movement', False)
    USE_SLOW_MOVEMENT = cfg_policy.lossconfig.get('use_slow_movement', False)
    BODY_ORI_WEIGHT = cfg_policy.lossconfig.get('body_ori_weight', 1.0)
    TARGET_DIST_WEIGHT = cfg_policy.lossconfig.get('target_dist_weight', 1.0)
    REPROJ_FACTOR = cfg_policy.modelconfig.get('reproj_factor', 1.0)
    USE_ROBUST_KLD = cfg_policy.lossconfig.get('use_robust_kld', True)
    USE_EARLY_STOP = cfg_policy.trainconfig.get('use_early_stop', False)

    """data"""
    with open(config_env.get_body_marker_path() + '/SSM2.json') as f:
        marker_ssm_67 = json.load(f)['markersets'][0]['indices']
    feet_markers = ['RHEE', 'RTOE', 'RRSTBEEF', 'LHEE', 'LTOE', 'LRSTBEEF']
    feet_marker_idx = [list(marker_ssm_67.keys()).index(marker_name) for marker_name in feet_markers]
    # ground data
    Rg = R.from_euler('xyz', np.array([0, 0, 0]), degrees=True)
    rotmat_g = Rg.as_matrix()
    # body motion data
    bm_path = config_env.get_body_model_path()
    host_name = config_env.get_host_name()
    vposer, _ = load_model(bm_path + '/vposer_v2_0', model_code=VPoser,
                           remove_words_in_model_weights='vp_model.', disable_grad=True)
    vposer.eval()
    vposer.to(device)

    """set GAMMA primitive networks"""
    genop_1frame_male = configure_model(cfg_1frame_male, args.gpu_index, args.random_seed)
    genop_1frame_female = configure_model(cfg_1frame_female, args.gpu_index, args.random_seed)
    genop_2frame_male = configure_model(cfg_2frame_male, args.gpu_index, args.random_seed)
    genop_2frame_female = configure_model(cfg_2frame_female, args.gpu_index, args.random_seed)

    """body model parsers"""
    pconfig_mp = {
        'n_batch': 10 * n_gens_1frame,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_mp = SMPLXParser(pconfig_mp)

    pconfig_2frame = {
        'n_batch': n_gens_1frame * 2,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_2frame = SMPLXParser(pconfig_2frame)

    pconfig_1frame = {
        'n_batch': 1 * n_gens_1frame,
        'device': device,
        'marker_placement': 'ssm2_67'
    }
    smplxparser_1frame = SMPLXParser(pconfig_1frame)

    """policy network and their optimizer"""
    policy_model = GAMMAPolicy(cfg_policy.modelconfig)
    policy_model.train()
    policy_model.to(device)
    optimizer = optim.AdamW(policy_model.parameters(),
                           lr=lr_po,
                           weight_decay=0.01)

    """main block for motion generation"""
    loss_names = ['Return_PPO_1f', 'Value_1f', 'Return_PPO_2f', 'Value_2f']
    epoch = 0
    if cfg_policy.trainconfig['resume_training']:
        ckp_list = sorted(glob.glob(os.path.join(cfg_policy.trainconfig['save_dir'], '*.ckp')),
                          key=os.path.getmtime)
        if len(ckp_list) > 0:
            checkpoint = torch.load(ckp_list[-1], map_location=device)
            policy_model.load_state_dict(checkpoint['model_state_dict'])
            for k, v in checkpoint['model_state_dict'].items():
                if torch.isnan(v).any():
                    print('nan:', k)
                else:
                    print('not nan:', k)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print('[INFO] --resuming training from {}'.format(ckp_list[-1]))

    """Train locomotion policy in random scenes"""
    dataset_path = ''
    scene_dir = 'data/scenes'
    scene_list_path = Path('data/scenes/random_scene_names.pkl')
    with open(scene_list_path, 'rb') as f:
        scene_list = pickle.load(f)
    print('#scene:', len(scene_list))
    batch_gen = BatchGeneratorSceneTrain(dataset_path='',
                                        scene_dir=scene_dir,
                                        scene_list=scene_list,
                                        scene_type='random_scene',
                                        body_model_path=bm_path,
                                        mp_list=None)

    init_pose_from_data = False
    data = batch_gen.next_body(sigma=GOAL_SIGMA, use_zero_pose=False, visualize=True, init_pose_from_data=init_pose_from_data, random_rotation_range=cfg.trainconfig.random_rotation_range, res=cfg.modelconfig.map_res, extent=cfg.modelconfig.map_extent,)
    data = batch_gen.next_body(sigma=GOAL_SIGMA, use_zero_pose=False, visualize=True, init_pose_from_data=init_pose_from_data, random_rotation_range=cfg.trainconfig.random_rotation_range, res=cfg.modelconfig.map_res, extent=cfg.modelconfig.map_extent,)
    data = batch_gen.next_body(sigma=GOAL_SIGMA, use_zero_pose=False, visualize=True, init_pose_from_data=init_pose_from_data, random_rotation_range=cfg.trainconfig.random_rotation_range, res=cfg.modelconfig.map_res, extent=cfg.modelconfig.map_extent,)
    data = batch_gen.next_body(sigma=GOAL_SIGMA, use_zero_pose=False, visualize=True, init_pose_from_data=init_pose_from_data, random_rotation_range=cfg.trainconfig.random_rotation_range, res=cfg.modelconfig.map_res, extent=cfg.modelconfig.map_extent,)
    last_epoch = epoch
    for epoch in tqdm(range(last_epoch, num_epochs)):
        # while epoch < num_epochs:
        """collect traj data from rollout for training"""
        traj_ppo_list = []
        if args.profile:
            t_start = time.time()
        for _ in range(num_envs_per_epoch):
            data = batch_gen.next_body(sigma=GOAL_SIGMA, use_zero_pose=False,
                                       init_pose_from_data=init_pose_from_data,
                                       random_rotation_range=cfg.trainconfig.random_rotation_range)
            with torch.no_grad():
                traj_ppo = rollout(data, max_depth=max_depth, epoch=epoch)
                if traj_ppo is None:
                    continue
            traj_ppo_list.append(traj_ppo)
        if args.profile:
            t_traj = time.time()
            print('time to collect traj:', t_traj - t_start)
        if epoch % 20 == 0:
            log_and_print('\n')

        use_map = 'map' in cfg_policy.modelconfig['body_repr']
        """update policy using the 1-frame data"""
        traj_states_1f_pool = torch.cat([traj_ppo['states'][0].detach() for traj_ppo in traj_ppo_list])  # [b,t,d]
        traj_local_map_1f_pool = None if not use_map else torch.cat(
            [traj_ppo['local_map'][0].detach() for traj_ppo in traj_ppo_list])  # [b,t,d]
        traj_z_1f_pool = torch.cat([traj_ppo['actions'][0].detach() for traj_ppo in traj_ppo_list])  # [b,d]
        traj_z_logprob_1f_pool = torch.cat(
            [traj_ppo['action_logprob'][0].detach() for traj_ppo in traj_ppo_list])  # [b,d]
        traj_gae_1f_pool = torch.cat([traj_ppo['gae'][0].detach() for traj_ppo in traj_ppo_list])  # [b,d]
        traj_returns_1f_pool = torch.cat([traj_ppo['returns'][0].detach() for traj_ppo in traj_ppo_list])  # [b,d]
        N_samples = traj_gae_1f_pool.shape[0]

        for _ in range(max_train_iter_1f):
            idx = 0
            rr = torch.randperm(N_samples)
            traj_states_1f_pool = traj_states_1f_pool[rr]
            traj_local_map_1f_pool = None if not use_map else traj_local_map_1f_pool[rr]
            traj_z_1f_pool = traj_z_1f_pool[rr]
            traj_z_logprob_1f_pool = traj_z_logprob_1f_pool[rr]
            traj_gae_1f_pool = traj_gae_1f_pool[rr]
            while idx < N_samples:
                traj_states_1f = traj_states_1f_pool[idx:idx + BATCH_SIZE]
                traj_local_map_1f = None if not use_map else traj_local_map_1f_pool[idx:idx + BATCH_SIZE]
                traj_z_1f = traj_z_1f_pool[idx:idx + BATCH_SIZE]
                traj_z_logprob_1f = traj_z_logprob_1f_pool[idx:idx + BATCH_SIZE]
                traj_gae_1f = traj_gae_1f_pool[idx:idx + BATCH_SIZE]
                traj_returns_1f = traj_returns_1f_pool[idx:idx + BATCH_SIZE]
                optimizer.zero_grad()
                if args.profile:
                    t1 = time.time()
                loss_1f, loss_items_1f, kld_thresh = calc_loss_policy_val(
                    traj_states_1f,
                    traj_local_map_1f,
                    traj_z_1f,
                    traj_z_logprob_1f,
                    traj_gae_1f,
                    traj_returns_1f)
                loss_1f.backward(retain_graph=False)
                if cfg.trainconfig.clip_gradient:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=cfg.trainconfig.max_norm, norm_type=2, error_if_nonfinite=False)
                optimizer.step()
                if args.profile:
                    t2 = time.time()
                    if idx == 0:
                        print('loss and optim per step:', t2 - t1)
                idx += BATCH_SIZE
            if kld_thresh.item() >= KLD_THRESH:
                # print('[info]-- reaches the kld thresh', kld_thresh.item())
                break

        """update policy using the 2-frame data"""
        traj_states_2f_pool = torch.cat([torch.cat(traj_ppo['states'][1:]).detach() for traj_ppo in traj_ppo_list if
                                         len(traj_ppo['states']) > 1])  # [b,t,d]
        traj_local_map_2f_pool = None if not use_map else torch.cat(
            [torch.cat(traj_ppo['local_map'][1:]).detach() for traj_ppo in traj_ppo_list if
             len(traj_ppo['local_map']) > 1])  # [b,t,d]
        traj_z_2f_pool = torch.cat([torch.cat(traj_ppo['actions'][1:]).detach() for traj_ppo in traj_ppo_list if
                                    len(traj_ppo['actions']) > 1])  # [b,d]
        traj_z_logprob_2f_pool = torch.cat(
            [torch.cat(traj_ppo['action_logprob'][1:]).detach() for traj_ppo in traj_ppo_list if
             len(traj_ppo['action_logprob']) > 1])  # [b,d]
        traj_gae_2f_pool = torch.cat([torch.cat(traj_ppo['gae'][1:]).detach() for traj_ppo in traj_ppo_list if
                                      len(traj_ppo['gae']) > 1])  # [b,d]
        traj_returns_2f_pool = torch.cat([torch.cat(traj_ppo['returns'][1:]).detach() for traj_ppo in traj_ppo_list if
                                          len(traj_ppo['returns']) > 1])  # [b,d]
        N_samples = traj_gae_2f_pool.shape[0]

        for _ in range(max_train_iter_2f):
            idx = 0
            rr = torch.randperm(N_samples)
            traj_states_2f_pool = traj_states_2f_pool[rr]
            traj_local_map_2f_pool = None if not use_map else traj_local_map_2f_pool[rr]
            traj_z_2f_pool = traj_z_2f_pool[rr]
            traj_z_logprob_2f_pool = traj_z_logprob_2f_pool[rr]
            traj_gae_2f_pool = traj_gae_2f_pool[rr]
            # print('2frame shuffle:', traj_gae_2f_pool.shape[0])
            while idx < N_samples:
                traj_states_2f = traj_states_2f_pool[idx:idx + BATCH_SIZE]
                traj_local_map_2f = None if not use_map else traj_local_map_2f_pool[idx:idx + BATCH_SIZE]
                traj_z_2f = traj_z_2f_pool[idx:idx + BATCH_SIZE]
                traj_z_logprob_2f = traj_z_logprob_2f_pool[idx:idx + BATCH_SIZE]
                traj_gae_2f = traj_gae_2f_pool[idx:idx + BATCH_SIZE]
                traj_returns_2f = traj_returns_2f_pool[idx:idx + BATCH_SIZE]
                optimizer.zero_grad()
                loss_2f, loss_items_2f, kld_thresh = calc_loss_policy_val(
                    traj_states_2f,
                    traj_local_map_2f,
                    traj_z_2f,
                    traj_z_logprob_2f,
                    traj_gae_2f,
                    traj_returns_2f)
                loss_2f.backward(retain_graph=False)
                if cfg.trainconfig.clip_gradient:
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=cfg.trainconfig.max_norm, norm_type=2, error_if_nonfinite=False)
                optimizer.step()
                idx += BATCH_SIZE
            if kld_thresh.item() >= KLD_THRESH:
                # print('[info]-- reaches the kld thresh', kld_thresh.item())
                break

        if args.profile:
            t_optim = time.time()
            print('time to train:', t_optim - t_traj)

        '''save the checkpoints'''
        save_per_x_ep = cfg_policy.trainconfig['saving_per_X_ep']
        if ((1 + epoch) % save_per_x_ep == 0):
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': policy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, cfg_policy.trainconfig['save_dir'] + "/last.ckp")

    # save final model to wandb
    art = wandb.Artifact("policy", type="model")
    art.add_file(cfg_policy.trainconfig['save_dir'] + "/last.ckp")
    wandb.log_artifact(art)

