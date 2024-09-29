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
import json
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

sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
from exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorReachOrient
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
                        states, bparam_seed, prev_betas, scene, obj_points=None, target_ori=None):
    [pred_markers,
     pred_params,
     act,
     act_log_prob,
     value] = motion_model.generate_ppo(policy_model,
                                        states.permute([1, 0, 2]),
                                        bparam_seed.permute([1, 0, 2]),
                                        obj_points=obj_points,
                                        target_ori=target_ori,
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

    target_ori_world = scene['target_forward_dir']  # [1, 3]
    target_ori_local = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1), target_ori_world[None, ...]) # [b, 1, 3]
    target_ori_local = target_ori_local.squeeze(1)  # [b, 3]
    target_ori_local = target_ori_local[:, :2] / torch.norm(target_ori_local[:, :2], dim=-1, keepdim=True)  # [b, 2], only care about xy projection on floor

    # import trimesh
    # markers = Y_l[0, 0].reshape(-1, 3)
    # forward_dir = torch.cat([target_ori_local[:1, :], torch.zeros(1, 1).to(markers.device)], dim=-1)  #[1, 3]
    # marker_points = trimesh.points.PointCloud(markers.detach().cpu().numpy(), colors=np.array([255, 0, 0, 255]))
    # nodes = np.zeros((1, 2, 3))
    # nodes[:, 1, :] = forward_dir.detach().cpu().numpy()
    # forward_arrow = trimesh.load_path(nodes)
    # trimesh.Scene([marker_points, forward_arrow]).show()

    return dist_xy, dist_xyz, fea_wpath, fea_marker_3d_n, fea_marker_h, target_ori_local


def get_rewards(bparams, vertices, joints, Y_l, R0, T0, wpath, scene, target_ori_local, last_step=False):
    t1 = time.time()
    pel_loc = joints[:, :, 0]
    Y_w = torch.einsum('bij,btpj->btpi', R0, Y_l) + T0[:, None, :, :]  # [b, t, p, 3]
    nb, nt = Y_l.shape[:2]
    h = 1 / 40
    pel_speed = torch.norm(pel_loc[:, 2:] - pel_loc[:, :-2], dim=-1) / (2 * h)  # [b, t-2]
    Y_l_speed = torch.norm(Y_l[:, 2:] - Y_l[:, :-2], dim=-1) / (2 * h)  # [b, t-2,P=67]
    # Y_l_speed = Y_l_speed.reshape(nb, -1)
    '''evaluate contact soft'''
    # # dist2gp = torch.abs(pel_loc[:,-1,-1]-pel_loc[:,0, -1])  # pelvis height not change, not suitable for sitting
    # dist2gp = (torch.abs(Y_w[:, :, :, 2].amin(dim=(1, 2))) - 0.05).clamp(min=0)  # feet on floor
    # # dist2skat = Y_l_speed.amin(dim=-1)
    # dist2skat = (Y_l_speed.amin(dim=-1) - 0.075).clamp(min=0)
    dist2gp = torch.abs(Y_w[:, :, feet_marker_idx, 2].amin(dim=-1) - 0.02).mean(dim=-1)  # feet on floor
    # dist2skat = Y_l_speed.amin(dim=-1)
    dist2skat = (Y_l_speed[:, :, feet_marker_idx].amin(dim=-1) - 0.075).clamp(min=0).mean(dim=-1)
    dist2run = (Y_l_speed[:, :, feet_marker_idx].amax(dim=-1) - 2).clamp(min=0).mean(dim=-1)
    r_floor = torch.exp(-dist2gp)
    r_skate = torch.exp(-dist2skat)
    r_contact_friction = torch.exp(-dist2gp) * torch.exp(-dist2skat)

    """penalize static motion"""
    r_nonstatic = (Y_l_speed[:, :, feet_marker_idx].amax(dim=(1, 2)) > 0.1).float()

    '''evaluate moving direction'''
    pt_wpath = wpath[-1:]
    target_wpath_l = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1), pt_wpath[None, ...] - T0)[:, :, :3]
    # r_target = target_wpath_l[:,0]-pel_loc[:, 0,:2]
    b_dir = pel_loc[:, -1, :3] - pel_loc[:, 0, :3]  # (b,3)
    w_dir = target_wpath_l[:, 0] - pel_loc[:, 0, :3]
    w_dir_n = w_dir / torch.norm(w_dir, dim=-1, keepdim=True).clip(min=1e-12)
    b_dir = b_dir / torch.norm(b_dir,dim=-1, keepdim=True).clip(min=1e-12)
    r_move_toward = (1+torch.einsum('bi,bi->b', w_dir_n, b_dir))/2

    body_pose = bparams[:, :, 6:69].reshape(nt * nb, -1)
    vp_embedding = vposer.encode(body_pose).loc
    latent_dim = vp_embedding.shape[-1]
    vp_norm = torch.norm(vp_embedding.reshape(nb, nt, -1), dim=-1).mean(dim=1)
    r_vp = torch.exp(-vp_norm / (latent_dim ** 0.5))

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

    # deviation from target orientation
    t_ori = target_ori_local  # body forward dir of SMPL is z axis
    r_target_ori = (torch.einsum('bi,bi->b', t_ori, b_ori) + 1) / 2.0
    # distance to target
    dist2target = torch.norm(target_wpath_l[:, :, :] - pel_loc[:, :, :], dim=-1, keepdim=False).clip(min=1e-12).amin(dim=-1)  # [b,]
    # r_target_dist = (1 - ((dist2target - cfg.trainconfig.goal_thresh).clip(min=1e-10)) ** 0.5)
    r_target_dist = torch.exp(-(dist2target - cfg.trainconfig.goal_thresh).clip(min=0))
    # check whether reach the goal within distance and orientation threshold
    use_early_stop = cfg.trainconfig.use_early_stop
    reach_goal = True if use_early_stop and ((r_target_ori > cfg.trainconfig.ori_thresh).float() + (dist2target < cfg.trainconfig.goal_thresh).float() == 2).any() else False
    if not last_step and not reach_goal and cfg.lossconfig.sparse_reward:
        r_target_ori = 0
        r_target_dist = 0
    r_target = r_target_dist * r_face_target

    # reward = r_contact+r_target+BODY_ORI_WEIGHT*r_ori+r_vp+r_slow
    reward = r_contact_friction * cfg.lossconfig.weight_contact_friction + r_vp * cfg.lossconfig.weight_vp + \
             r_floor * cfg.lossconfig.weight_floor + r_skate * cfg.lossconfig.weight_skate + \
             r_move_toward * cfg.lossconfig.weight_move_toward + r_face_target * cfg.lossconfig.weight_face_target + \
             r_target_dist * cfg.lossconfig.weight_target_dist + \
             r_target_ori * cfg.lossconfig.weight_target_ori + \
             r_target * cfg.lossconfig.weight_target + \
             r_nonstatic * cfg.lossconfig.weight_nonstatic
    # reward = reward + r_floor + r_pelvis_vel
    if last_step or reach_goal or not cfg.lossconfig.sparse_reward:
        wandb.log({
            'r_target_dist': r_target_dist.mean(),
            'r_target_ori': r_target_ori.mean(),
        })
    wandb.log({
        'r_contact_friction': r_contact_friction.mean(),
        'r_nonstatic': r_nonstatic.mean(),
        'r_floor': r_floor.mean(),
        'r_skate': r_skate.mean(),
        'r_vp': r_vp.mean(),
        'r_move_toward': r_move_toward.mean(),
        'r_face_target': r_face_target.mean(),
        'reward': reward.mean(),
    })
    return reward, reach_goal


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

    distxy, dist, fea_wpath, fea_marker, fea_marker_h, target_ori_local = get_feature(marker_seed, pelvis_loc[:, :t_his],
                                                                                     R0, T0, wpath[-1:], scene)
    # if np.any(dist.detach().cpu().numpy() < GOAL_THRESH):
    #     warnings.warn('[warning] the current target is too close to the starting location!')
    #     return None
    states = torch.cat([marker_seed, fea_marker], dim=-1) if 'condi' in cfg_policy.modelconfig['body_repr'] else marker_seed
    obj_points_local = None
    if 'pointcloud' in cfg_policy.modelconfig['body_repr']:
        obj_points_local = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1),
                                        scene['obj_points'][None, ...] - T0)  # [b, p, 3]
    # generate markers and regress to body params
    [pred_markers, pred_params,
     act, act_log_prob, value] = gen_motion_one_step(
        motion_model, policy_model,
        states, body_param_seed, prev_betas, scene,
        obj_points=obj_points_local,
        target_ori=target_ori_local if 'orient' in cfg_policy.modelconfig['body_repr'] else None,
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
    traj_rewards, reach_goal = get_rewards(pred_params, pred_vertices, pred_joints, pred_marker_b, R0, T0, wpath, scene, target_ori_local)
    traj_states = states  # [b,t,d]
    traj_target_ori = target_ori_local # [b, 2]
    traj_act = act  # [b,d]
    traj_act_logprob = act_log_prob  # [b,]
    traj_value = value  # [b,]
    return [[traj_states, traj_target_ori, traj_act, traj_act_logprob, traj_rewards, traj_value],
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
        xb=body_param_seed.reshape(nb * nt, -1),
        to_numpy=False,
        inplace=False).reshape(nb, nt, -1)

    marker_seed = torch.einsum('bij,btpj->btpi', R_.permute(0, 2, 1), prev_markers_b[:, -t_his:] - T_[..., None, :])
    pel_loc_seed = torch.einsum('bij,btj->bti', R_.permute(0, 2, 1), prev_pelvis_loc[:, -t_his:] - T_)
    distxy, dist, fea_wpath, fea_marker, fea_marker_h, target_ori_local = get_feature(marker_seed, pel_loc_seed, R0, T0,
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
    obj_points_local = None
    if 'pointcloud' in cfg_policy.modelconfig['body_repr']:
        obj_points_local = torch.einsum('bij,btj->bti', R0.permute(0, 2, 1),
                                        scene['obj_points'][None, ...] - T0)  # [b, p, 3]

    if args.profile:
        t2 = time.time()
        print('time for calc state:', t2 - t1)
    # generate markers and regress to body params
    [pred_markers, pred_params,
     act, act_log_prob, value] = gen_motion_one_step(
        motion_model, policy_model,
        states, body_param_seed, prev_betas, scene,
        obj_points=obj_points_local,
        target_ori=target_ori_local if 'orient' in cfg_policy.modelconfig['body_repr'] else None,
    )
    if args.profile:
        t3 = time.time()
        print('time for model inference', t3 - t2)
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
    if args.profile:
        t4 = time.time()
        print('time for SMPLX inference', t4 - t3)
        t4 = time.time()
    traj_rewards, reach_goal = get_rewards(pred_params, pred_vertices, pred_joints, pred_marker_b, R0, T0,
                               wpath, scene, target_ori_local,
                               last_step=last_step)
    if args.profile:
        t5 = time.time()
        print('time for cal reward', t5 - t4)
    traj_states = states  # [b,t,d]
    traj_target_ori = target_ori_local # [b, 2]
    traj_act = act  # [b,d]
    traj_act_logprob = act_log_prob  # [b,]
    traj_value = value  # [b,]

    return [[traj_states, traj_target_ori, traj_act, traj_act_logprob, traj_rewards, traj_value],
            dist2target, reach_goal,
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
        outmps_nodes = {'motion': [], 'wpath': wpath.detach().cpu().numpy(), 'target_orient': scene['target_orient'].detach().cpu().numpy()}
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
                'target_ori': [],
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
            data_traj, dist2target_, reach_goal, data_mp = mp_new
            dist2target_record.append(dist2target_)
            outmps.append(data_mp)
            for i, key in enumerate(traj_ppo.keys()):
                traj_ppo[key].append(data_traj[i])
            if reach_goal:
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
    if epoch % 20 == 0:
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
                         traj_target_ori,
                         traj_z,
                         traj_z_logprob,
                         traj_gae,
                         traj_returns):
    '''forward pass of the policy value model'''
    z_mu, z_logvar, value = policy_model(traj_states.permute([1, 0, 2]), target_ori=traj_target_ori)
    z_var = torch.exp(z_logvar.clamp(policy_model.min_logvar, policy_model.max_logvar))
    act_distrib_c = Normal(z_mu, z_var ** 0.5)  # batched distribution
    act_distrib = Independent(act_distrib_c, 1)

    '''calculate ppo'''
    traj_z_logprob_new = act_distrib.log_prob(traj_z)  # size=(b=1, )
    policy_ratio = torch.exp(traj_z_logprob_new - traj_z_logprob)
    clipped_ratio = policy_ratio.clamp(1 - PPO_CLIP_VAL,
                                       1 + PPO_CLIP_VAL)
    clipped_loss = clipped_ratio * traj_gae
    full_loss = policy_ratio * traj_gae
    loss_ppo = -torch.min(full_loss, clipped_loss).mean()

    '''KLD like in Motion Primitive VAE'''
    loss_kld = 0.5 * torch.mean(-1 - z_logvar + z_mu.pow(
        2) + z_logvar.exp())  # batchmean is mathematically correct. https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    if USE_ROBUST_KLD:
        loss_kld = (torch.sqrt(1 + loss_kld ** 2) - 1)
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


@hydra.main(version_base=None, config_path="../exp_GAMMAPrimitive/cfg", config_name="MPVAEPolicy_babel_locomotion")
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
    optimizer = optim.Adam(policy_model.parameters(),
                           lr=lr_po)

    """main block for motion generation"""
    loss_names = ['Return_PPO_1f', 'Value_1f', 'Return_PPO_2f', 'Value_2f']
    epoch = 0
    if cfg_policy.trainconfig['resume_training']:
        ckp_list = sorted(glob.glob(os.path.join(cfg_policy.trainconfig['save_dir'], '*.ckp')),
                          key=os.path.getmtime)
        if len(ckp_list) > 0:
            checkpoint = torch.load(ckp_list[-1], map_location=device)
            policy_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print('[INFO] --resuming training from {}'.format(ckp_list[-1]))

    # print('save trajs at:', os.path.join(cfg_policy.trainconfig['save_dir'], args.exp_name, 'results'))

    batch_gen = BatchGeneratorReachOrient(dataset_path='data',
                                             body_model_path=bm_path)
    # data = batch_gen.next_body(sigma=GOAL_SIGMA, use_zero_pose=False, visualize=True)
    # data = batch_gen.next_body(sigma=GOAL_SIGMA, use_zero_pose=False, visualize=True)
    # data = batch_gen.next_body(sigma=GOAL_SIGMA, use_zero_pose=False, visualize=True)
    # num_envs_per_epoch = 8
    last_epoch = epoch
    for epoch in tqdm(range(last_epoch, num_epochs)):
        # while epoch < num_epochs:
        """collect traj data from rollout for training"""
        traj_ppo_list = []
        if args.profile:
            t_start = time.time()
        for _ in range(num_envs_per_epoch):
            data = batch_gen.next_body(sigma=GOAL_SIGMA, use_zero_pose=False)
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

        use_orient = 'orient' in cfg_policy.modelconfig['body_repr']
        """update policy using the 1-frame data"""
        traj_states_1f_pool = torch.cat([traj_ppo['states'][0].detach() for traj_ppo in traj_ppo_list])  # [b,t,d]
        traj_target_ori_1f_pool = None if not use_orient else torch.cat(
            [traj_ppo['target_ori'][0].detach() for traj_ppo in traj_ppo_list])  # [b,t,d]
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
            traj_target_ori_1f_pool = None if not use_orient else traj_target_ori_1f_pool[rr]
            traj_z_1f_pool = traj_z_1f_pool[rr]
            traj_z_logprob_1f_pool = traj_z_logprob_1f_pool[rr]
            traj_gae_1f_pool = traj_gae_1f_pool[rr]
            while idx < N_samples:
                traj_states_1f = traj_states_1f_pool[idx:idx + BATCH_SIZE]
                traj_target_ori_1f = None if not use_orient else traj_target_ori_1f_pool[idx:idx + BATCH_SIZE]
                traj_z_1f = traj_z_1f_pool[idx:idx + BATCH_SIZE]
                traj_z_logprob_1f = traj_z_logprob_1f_pool[idx:idx + BATCH_SIZE]
                traj_gae_1f = traj_gae_1f_pool[idx:idx + BATCH_SIZE]
                traj_returns_1f = traj_returns_1f_pool[idx:idx + BATCH_SIZE]
                optimizer.zero_grad()
                if args.profile:
                    t1 = time.time()
                loss_1f, loss_items_1f, kld_thresh = calc_loss_policy_val(
                    traj_states_1f,
                    traj_target_ori_1f,
                    traj_z_1f,
                    traj_z_logprob_1f,
                    traj_gae_1f,
                    traj_returns_1f)
                loss_1f.backward(retain_graph=False)
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
        traj_target_ori_2f_pool = None if not use_orient else torch.cat(
            [torch.cat(traj_ppo['target_ori'][1:]).detach() for traj_ppo in traj_ppo_list if
             len(traj_ppo['target_ori']) > 1])  # [b,t,d]
        traj_z_2f_pool = torch.cat([torch.cat(traj_ppo['actions'][1:]).detach() for traj_ppo in traj_ppo_list if
                                    len(traj_ppo['actions']) > 1])  # [b,d]
        traj_z_logprob_2f_pool = torch.cat(
            [torch.cat(traj_ppo['action_logprob'][1:]).detach() for traj_ppo in traj_ppo_list if
             len(traj_ppo['action_logprob']) > 1])  # [b,d]
        traj_gae_2f_pool = torch.cat([torch.cat(traj_ppo['gae'][1:]).detach() for traj_ppo in traj_ppo_list if
                                      len(traj_ppo['gae']) > 1])  # [b,d]
        traj_returns_2f_pool = torch.cat([torch.cat(traj_ppo['returns'][1:]).detach() for traj_ppo in traj_ppo_list if
                                          len(traj_ppo['returns']) > 1])  # [b,d]

        for _ in range(max_train_iter_2f):
            idx = 0
            rr = torch.randperm(N_samples)
            traj_states_2f_pool = traj_states_2f_pool[rr]
            traj_target_ori_2f_pool = None if not use_orient else traj_target_ori_2f_pool[rr]
            traj_z_2f_pool = traj_z_2f_pool[rr]
            traj_z_logprob_2f_pool = traj_z_logprob_2f_pool[rr]
            traj_gae_2f_pool = traj_gae_2f_pool[rr]
            while idx < N_samples:
                traj_states_2f = traj_states_2f_pool[idx:idx + BATCH_SIZE]
                traj_target_ori_2f = None if not use_orient else traj_target_ori_2f_pool[idx:idx + BATCH_SIZE]
                traj_z_2f = traj_z_2f_pool[idx:idx + BATCH_SIZE]
                traj_z_logprob_2f = traj_z_logprob_2f_pool[idx:idx + BATCH_SIZE]
                traj_gae_2f = traj_gae_2f_pool[idx:idx + BATCH_SIZE]
                traj_returns_2f = traj_returns_2f_pool[idx:idx + BATCH_SIZE]
                optimizer.zero_grad()
                loss_2f, loss_items_2f, kld_thresh = calc_loss_policy_val(
                    traj_states_2f,
                    traj_target_ori_2f,
                    traj_z_2f,
                    traj_z_logprob_2f,
                    traj_gae_2f,
                    traj_returns_2f)
                loss_2f.backward(retain_graph=False)
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
            # torch.save({
            #     'epoch': epoch + 1,
            #     'model_state_dict': policy_model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            # }, cfg_policy.trainconfig['save_dir'] + "/epoch-" + str(epoch + 1) + ".ckp")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': policy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, cfg_policy.trainconfig['save_dir'] + "/last.ckp")

        # epoch += 1

    # save final model to wandb
    art = wandb.Artifact("policy", type="model")
    art.add_file(cfg_policy.trainconfig['save_dir'] + "/last.ckp")
    wandb.log_artifact(art)