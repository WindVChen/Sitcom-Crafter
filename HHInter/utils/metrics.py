import numpy as np
import torch
from HHInter.global_path import *
from scipy import linalg
import smplx
import json
from HHInter.models.losses import GeneralContactLoss, CustomLoss, GeometricLoss
from HHInter.models.blocks import MoshRegressor

emb_scale = 6

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist


def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    activations = activations * emb_scale
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    activation = activation * emb_scale
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm((activation[first_indices] - activation[second_indices]) / 2, axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


class CalculatePhysics(object):
    def __init__(self):
        self.body_regressor = MoshRegressor().cuda()
        self.body_regressor.load_state_dict(
            torch.load(get_smplx_body_regressor_checkpoint_path(), map_location='cuda')['model_state_dict'])
        self.body_regressor.eval()

        with open(get_SSM_SMPLX_body_marker_path()) as f:
            markerdict = json.load(f)['markersets'][0]['indices']
        self.markers = list(markerdict.values())

        # Construct sdf points
        sdf_points_extents = 3.  # InterGen dataset motion maximum extent is 6.3149.
        ceiling_height = 3.
        sdf_points_res = 128

        x = torch.linspace(-sdf_points_extents, sdf_points_extents, sdf_points_res)
        y = torch.linspace(-sdf_points_extents, sdf_points_extents, sdf_points_res)
        z = torch.linspace(-ceiling_height, ceiling_height, sdf_points_res)

        x, y, z = torch.meshgrid(x, y, z)
        self.sdf_coord = torch.stack([x, y, z], dim=-1).permute(3, 0, 1, 2).cuda()

    def generate_src_mask(self, T, length):
        B = length.shape[0]
        src_mask = torch.ones(B, T, 2)
        for p in range(2):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask

    @torch.no_grad()
    def calculate_human_physics(self, batch_data, is_gt=True):
        name, text, motion1_all, motion2_all, motion_lens, _, _, _, _, _, _ = batch_data

        all_B, all_T = motion1_all.shape[:2]

        # The first frame is for condition.
        if is_gt:
            motion1_all = motion1_all.reshape(all_B, all_T, 68, 3)
            motion1_all = motion1_all[:, :, :67, :] + motion1_all[:, :, 67:, :]
            motion1_all = motion1_all.reshape(all_B, all_T, -1)

            motion2_all = motion2_all.reshape(all_B, all_T, 68, 3)
            motion2_all = motion2_all[:, :, :67, :] + motion2_all[:, :, 67:, :]
            motion2_all = motion2_all.reshape(all_B, all_T, -1)
            motion1_all = motion1_all[:, 1:].type(torch.float32).cuda()
            motion2_all = motion2_all[:, 1:].type(torch.float32).cuda()
            all_T -= 1
        else:
            motion1_all = motion1_all.type(torch.float32).cuda()
            motion2_all = motion2_all.type(torch.float32).cuda()

        all_loss = 0

        # Sub B.
        B = 8
        skip_step = 10

        for i in range(0, all_B, B):
            motion1 = motion1_all[i:i + B][:, ::skip_step]
            motion2 = motion2_all[i:i + B][:, ::skip_step]

            seq_mask = self.generate_src_mask(all_T, motion_lens[i:i + B]).reshape(B, all_T, -1, 1).cuda()
            seq_mask = seq_mask[:, ::skip_step]

            # Reset T.
            T = motion1.shape[1]

            self.smplx_model = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplx',
                                            gender='neutral', ext='pkl',
                                            num_betas=10,
                                            num_pca_comps=12,
                                            batch_size=B * T
                                            ).cuda()
            self.smplx_model.eval()

            params_A = self.body_regressor(motion1[:, :, :67 * 3].reshape(-1, 67 * 3), B, T, seq_mask[..., 0, :])
            bparam = {}
            bparam['transl'] = params_A[:, :3]
            bparam['global_orient'] = params_A[:, 3:6]
            bparam['body_pose'] = params_A[:, 6:69]
            # bparam['left_hand_pose'] = params_A[:, 69:81]
            # bparam['right_hand_pose'] = params_A[:, 81:93]
            bparam['betas'] = params_A[:, 93:103]
            vertices_A = self.smplx_model(return_verts=True, **bparam).vertices

            params_B = self.body_regressor(motion2[:, :, :67 * 3].reshape(-1, 67 * 3), B, T, seq_mask[..., 0, :])
            bparam = {}
            bparam['transl'] = params_B[:, :3]
            bparam['global_orient'] = params_B[:, 3:6]
            bparam['body_pose'] = params_B[:, 6:69]
            # bparam['left_hand_pose'] = params_B[:, 69:81]
            # bparam['right_hand_pose'] = params_B[:, 81:93]
            bparam['betas'] = params_B[:, 93:103]
            vertices_B = self.smplx_model(return_verts=True, **bparam).vertices

            humanpeneloss_manager = GeneralContactLoss(body_model_utils_folder=get_human_penetration_essentials_path())
            humanpeneloss_manager.weight = 1
            humanpeneloss_manager.forward(seq_mask, torch.ones(B).cuda(), v1=vertices_A.view(B, T, -1, 3),
                                          v2=vertices_B.view(B, T, -1, 3), factor=1)

            all_loss += humanpeneloss_manager.losses['humanpenetration'].cpu().numpy()

        all_loss /= (all_B // B)

        return all_loss

    @torch.no_grad()
    def calculate_scene_penet(self, batch_data, is_gt=True):
        name, text, all_motion1, all_motion2, all_motion_lens, _, _, _, _, _, all_sdf_points = batch_data

        all_loss_A, all_loss_B = 0, 0

        sub_B = 16

        for i in range(0, len(text), sub_B):
            motion1 = all_motion1[i:i + sub_B]
            motion2 = all_motion2[i:i + sub_B]
            sdf_points = all_sdf_points[i:i + sub_B].type(torch.float32).cuda()
            motion_lens = all_motion_lens[i:i + sub_B]


            B, T = motion1.shape[:2]

            # The first frame is for condition.
            if is_gt:
                motion1 = motion1.reshape(B, T, 68, 3)
                motion1 = motion1[:, :, :67, :] + motion1[:, :, 67:, :]
                motion1 = motion1.reshape(B, T, -1)

                motion2 = motion2.reshape(B, T, 68, 3)
                motion2 = motion2[:, :, :67, :] + motion2[:, :, 67:, :]
                motion2 = motion2.reshape(B, T, -1)
                sdf_points = torch.cat([self.sdf_coord.unsqueeze(0).expand(motion1.shape[0], -1, -1, -1, -1),
                                        sdf_points.type(torch.float32).cuda()], dim=1)
                motion1 = motion1[:, 1:].type(torch.float32).cuda()
                motion2 = motion2[:, 1:].type(torch.float32).cuda()
                T -= 1
            else:
                motion1 = motion1.type(torch.float32).cuda()
                motion2 = motion2.type(torch.float32).cuda()

            seq_mask = self.generate_src_mask(T, motion_lens).reshape(B, T, -1, 1).cuda()

            loss_a_manager = CustomLoss("A")
            loss_a_manager.weights["weight_pene"] = 1
            loss_a_manager.forward(motion1[..., :], sdf_points, None, None, seq_mask[..., 0, :],
                                   torch.ones(B).cuda())

            loss_b_manager = CustomLoss("B")
            loss_b_manager.weights["weight_pene"] = 1
            loss_b_manager.forward(motion2[..., :], sdf_points, None, None, seq_mask[..., 0, :],
                                   torch.ones(B).cuda())

            all_loss_A += loss_a_manager.losses['A_r_pene'].cpu().numpy()
            all_loss_B += loss_b_manager.losses['B_r_pene'].cpu().numpy()

        all_loss_A /= (len(text) // sub_B)
        all_loss_B /= (len(text) // sub_B)

        return all_loss_A, all_loss_B

    @torch.no_grad()
    def calculate_foot_physics(self, batch_data, is_gt=True):
        name, text, motion1, motion2, motion_lens, _, motion_R, motion_T, _, feet_height_thresh, _ = batch_data

        B, T = motion1.shape[:2]

        # The first frame is for condition.
        if is_gt:
            motion1 = motion1.reshape(B, T, 68, 3)
            motion1 = motion1[:, :, :67, :] + motion1[:, :, 67:, :]
            motion1 = motion1.reshape(B, T, -1)

            motion2 = motion2.reshape(B, T, 68, 3)
            motion2 = motion2[:, :, :67, :] + motion2[:, :, 67:, :]
            motion2 = motion2.reshape(B, T, -1)
            motion1 = motion1[:, 1:].type(torch.float32).cuda()
            motion2 = motion2[:, 1:].type(torch.float32).cuda()
            T -= 1
        else:
            motion1 = motion1.type(torch.float32).cuda()
            motion2 = motion2.type(torch.float32).cuda()

        seq_mask = self.generate_src_mask(T, motion_lens).reshape(B, T, -1, 1).cuda()

        motion_R = motion_R.type(torch.float32)
        motion_T = motion_T.type(torch.float32)

        loss_a_manager_geo = GeometricLoss("l2", 67, "Geo_A")
        loss_a_manager_geo.forward(motion1[..., :], motion1[..., :], seq_mask[..., 0, :],
                                   torch.ones(B).cuda(), motion_R[..., :3].cuda(), motion_T[..., :3].cuda(), None,
                                   feet_height_thresh[:, :6].cuda(), is_eval=True)

        loss_b_manager_geo = GeometricLoss("l2", 67, "Geo_B")
        loss_b_manager_geo.forward(motion2[..., :], motion2[..., :], seq_mask[..., 0, :],
                                   torch.ones(B).cuda(), motion_R[..., 3:].cuda(), motion_T[..., 3:].cuda(), None,
                                   feet_height_thresh[:, 6:].cuda(), is_eval=True)

        return loss_a_manager_geo.losses['FC_slide_Geo_A'].cpu().numpy() + loss_b_manager_geo.losses['FC_slide_Geo_B'].cpu().numpy(), \
            loss_a_manager_geo.losses['FC_pene_Geo_A'].cpu().numpy() + loss_b_manager_geo.losses['FC_pene_Geo_B'].cpu().numpy()