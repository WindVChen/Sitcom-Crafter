import sys

import torch

sys.path.append(sys.path[0]+"/../")

from HHInter.global_path import *
import smplx
import json
from HHInter.models.losses import GeneralContactLoss, GeometricLoss
from collections import OrderedDict
from HHInter.models import *
import pickle
import tqdm


def calc_sdf(vertices, sdf_dict):
    sdf_centroid = sdf_dict['centroid'].cuda()
    sdf_scale = sdf_dict['scale']
    sdf_grids = sdf_dict['grid'].cuda()

    batch_size, seq_len, num_vertices, _ = vertices.shape
    vertices = vertices.reshape(1, -1, 3)  # [B, V, 3]
    vertices = (vertices - sdf_centroid) / sdf_scale  # convert to [-1, 1]
    sdf_values = F.grid_sample(sdf_grids,
                               vertices[:, :, [2, 1, 0]].view(batch_size, seq_len * num_vertices, 1, 1, 3),
                               # [2,1,0] permute because of grid_sample assumes different dimension order, see below
                               padding_mode='border',
                               align_corners=True
                               # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                               ).reshape(batch_size, seq_len, num_vertices)

    return sdf_values

class CustomLoss(nn.Module):
    def __init__(self, name):
        super(CustomLoss, self).__init__()
        self.name = name

        self.weights = {}
        self.weights["weight_pene"] = 1

        self.losses = {}

    def mix_masked_mse(self, loss, mask, batch_mask):
        loss = (loss * mask).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    def forward(self, Y_l, scene_sdf, motion_R, motion_T, mask, timestep_mask):
        Y_l = Y_l.view(Y_l.shape[0], Y_l.shape[1], 67, 3)
        vertices_l = Y_l

        # penetration reward
        sdf_values = calc_sdf(vertices_l, scene_sdf) # [b, t*p]

        sdf_values[sdf_values >= 0] = 1
        sdf_values[sdf_values < 0] = -1

        """also consider penetration with floor (not directly used counted number, which is unstable)"""
        negative_values = sdf_values * (sdf_values < 0)

        if vertices_l.shape[2] != 67:
            l_penetration = -negative_values.sum(dim=-1, keepdim=True)
        else:
            l_penetration = -negative_values.sum(dim=-1, keepdim=True)

        l_penetration = self.mix_masked_mse(l_penetration, mask, timestep_mask)

        loss = {
            self.name + '_r_pene': l_penetration * self.weights["weight_pene"],
        }
        self.losses.update(loss)


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
            motion1_all = motion1_all[:, 1:].type(torch.float32).cuda()
            motion2_all = motion2_all[:, 1:].type(torch.float32).cuda()
            all_T -= 1
        else:
            motion1_all = motion1_all.type(torch.float32).cuda()
            motion2_all = motion2_all.type(torch.float32).cuda()

        all_loss = 0

        # Sub B.
        B = 1
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

        sub_B = 1

        for i in range(0, len(all_motion1), sub_B):
            motion1 = all_motion1[i:i + sub_B]
            motion2 = all_motion2[i:i + sub_B]
            sdf_points = all_sdf_points
            motion_lens = all_motion_lens[i:i + sub_B]


            B, T = motion1.shape[:2]

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

        all_loss_A /= (len(all_motion1) // sub_B)
        all_loss_B /= (len(all_motion1) // sub_B)

        return all_loss_A, all_loss_B

    @torch.no_grad()
    def calculate_foot_physics(self, batch_data, is_gt=True):
        name, text, motion1, motion2, motion_lens, _, motion_R, motion_T, _, feet_height_thresh, _ = batch_data

        B, T = motion1.shape[:2]

        # The first frame is for condition.
        if is_gt:
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
                                   torch.ones(B).cuda(), motion_R.cuda(), motion_T.cuda(), None,
                                   feet_height_thresh[:, :6].cuda(), is_eval=True)

        loss_b_manager_geo = GeometricLoss("l2", 67, "Geo_B")
        loss_b_manager_geo.forward(motion2[..., :], motion2[..., :], seq_mask[..., 0, :],
                                   torch.ones(B).cuda(), motion_R.cuda(), motion_T.cuda(), None,
                                   feet_height_thresh[:, 6:].cuda(), is_eval=True)

        return loss_a_manager_geo.losses['FC_slide_Geo_A'].cpu().numpy() + loss_b_manager_geo.losses['FC_slide_Geo_B'].cpu().numpy(), \
            loss_a_manager_geo.losses['FC_pene_Geo_A'].cpu().numpy() + loss_b_manager_geo.losses['FC_pene_Geo_B'].cpu().numpy()

def evaluate_matching_score(motion_files, file):
    motion_files = os.listdir(motion_files)
    foot_slide_dict = OrderedDict({})
    foot_penetration_dict = OrderedDict({})
    scene_penetration_dict = OrderedDict({})
    human_penetration_dict = OrderedDict({})

    # with open(log_file, 'w') as file:
    file = open(log_file, 'w')

    physics_metric = CalculatePhysics()

    # print(motion_loaders.keys())
    all_size = len(motion_files)
    batch_size = 1
    foot_slide = 0
    foot_penetration = 0
    scene_penetration = 0
    human_penetration = 0
    scene_A = 0
    scene_B = 0
    # print(motion_loader_name)
    motion_loader_name = "story"
    flag = 'smplx'
    with torch.no_grad():
        for mot in tqdm.tqdm(motion_files):
            scene_name = mot.split("_")[3] + "_" + mot.split("_")[4]
            if os.path.exists(os.path.join(pickle_file_root, mot, "smplx-bvh", "person1.pkl")):
                person1 = os.path.join(pickle_file_root, mot, "smplx-bvh", "person1.pkl")
                person2 = os.path.join(pickle_file_root, mot, "smplx-bvh", "person2.pkl")
                '''get markers'''
                with open(get_SSM_SMPLX_body_marker_path()) as f:
                    markerdict = json.load(f)['markersets'][0]['indices']
                markers = list(markerdict.values())
            else:
                person1 = os.path.join(pickle_file_root, mot, "smplh-bvh", "person1.pkl")
                person2 = os.path.join(pickle_file_root, mot, "smplh-bvh", "person2.pkl")
                flag = 'smplh'
                '''get markers'''
                with open(get_SSM_SMPL_body_marker_path()) as f:
                    markerdict = json.load(f)['markersets'][0]['indices']
                markers = list(markerdict.values())
            # load person motion
            person1_motion = pickle.load(open(person1, "rb"))
            person2_motion = pickle.load(open(person2, "rb"))

            motion_lens = min(len(person1_motion), len(person2_motion))
            person1_motion = torch.from_numpy(person1_motion[:motion_lens])
            person2_motion = torch.from_numpy(person2_motion[:motion_lens])

            sdf_path = os.path.join(get_program_root_path(), "Sitcom-Crafter/HSInter/data/replica", scene_name, "sdf", "scene_sdf.pkl")
            scene_sdf = pickle.load(open(sdf_path, "rb"))

            # identity matrix
            motion_R = torch.eye(3).unsqueeze(0)
            motion_T = torch.zeros(3).unsqueeze(0).unsqueeze(0)

            smplx_model = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type=flag,
                                       gender='neutral', ext='pkl',
                                       num_betas=10,
                                       num_pca_comps=12,
                                       batch_size=motion_lens  # 300 is time sequence
                                       )
            bparam = {}
            bparam['transl'] = person1_motion[:, :3]
            bparam['global_orient'] = person1_motion[:, 3:6]
            bparam['body_pose'] = person1_motion[:, 6:69]
            bparam['betas'] = person1_motion[:, -10:]
            vertices = smplx_model(return_verts=True, **bparam).vertices
            motion1 = vertices[:, markers].view(1, len(person1_motion), -1, 3)

            bparam = {}
            bparam['transl'] = person2_motion[:, :3]
            bparam['global_orient'] = person2_motion[:, 3:6]
            bparam['body_pose'] = person2_motion[:, 6:69]
            bparam['betas'] = person2_motion[:, -10:]
            vertices = smplx_model(return_verts=True, **bparam).vertices
            motion2 = vertices[:, markers].view(1, len(person2_motion), -1, 3)

            smplx_model = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplx',
                                       gender='neutral', ext='pkl',
                                       num_betas=10,
                                       num_pca_comps=12,
                                       batch_size=1  # 300 is time sequence
                                       )
            smpl_1 = smplx_model(return_vertices=True, betas=person1_motion[:, -10:][[0]])
            marker_height_1 = (smpl_1.vertices[0, markers][[47, 60, 55, 16, 30, 25]][:, 1] - smpl_1.joints[0, :, 1].min()).detach()
            smpl_2 = smplx_model(return_vertices=True, betas=person1_motion[:, -10:][[0]])
            marker_height_2 = (smpl_2.vertices[0, markers][[47, 60, 55, 16, 30, 25]][:, 1] - smpl_2.joints[0, :, 1].min()).detach()

            feet_height_thresh = torch.cat([marker_height_1, marker_height_2], dim=-1).unsqueeze(0)


            batch_data = [None, None, motion1, motion2, torch.tensor([motion_lens]), None, motion_R, motion_T, None, feet_height_thresh, scene_sdf]

            batch_marker_format = batch_data
            slide, f_pene = physics_metric.calculate_foot_physics(batch_marker_format, is_gt=False)
            s_pene_A, s_pene_B = physics_metric.calculate_scene_penet(batch_marker_format, is_gt='ground truth' in motion_loader_name)
            h_pene = physics_metric.calculate_human_physics(batch_marker_format, is_gt='ground truth' in motion_loader_name)

            print(f'---> [{mot}] Foot sliding: {slide:.4f}, Foot penetration: {f_pene:.4f}, '
                  f'Scene penetration: {s_pene_A + s_pene_B:.4f}, Human penetration: {h_pene:.4f}')
            print(f'---> [{mot}] Foot sliding: {slide:.4f}, Foot penetration: {f_pene:.4f}, '
                  f'Scene penetration: {s_pene_A + s_pene_B:.4f}, Human penetration: {h_pene:.4f}', file=file, flush=True)

            foot_slide += slide
            foot_penetration += f_pene
            scene_penetration += s_pene_A + s_pene_B
            scene_A += s_pene_A
            scene_B += s_pene_B
            human_penetration += h_pene

            # break

        foot_slide_dict[motion_loader_name] = foot_slide / (all_size // batch_size)
        foot_penetration_dict[motion_loader_name] = foot_penetration / (all_size // batch_size)
        scene_penetration_dict[motion_loader_name] = scene_penetration / (all_size // batch_size)
        human_penetration_dict[motion_loader_name] = human_penetration / (all_size // batch_size)

        scene_A = scene_A / (all_size // batch_size)
        scene_B = scene_B / (all_size // batch_size)

    print(f'---> [{motion_loader_name}] Foot sliding: {foot_slide_dict[motion_loader_name]:.4f}')
    print(f'---> [{motion_loader_name}] Foot sliding: {foot_slide_dict[motion_loader_name]:.4f}', file=file, flush=True)

    print(f'---> [{motion_loader_name}] Foot penetration: {foot_penetration_dict[motion_loader_name]:.4f}')
    print(f'---> [{motion_loader_name}] Foot penetration: {foot_penetration_dict[motion_loader_name]:.4f}', file=file, flush=True)

    print(f'---> [{motion_loader_name}] Scene penetration: {scene_penetration_dict[motion_loader_name]:.4f}')
    print(f'---> [{motion_loader_name}] Scene penetration: {scene_penetration_dict[motion_loader_name]:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] Scene penetration A: {scene_A:.4f}', file=file, flush=True)
    print(f'---> [{motion_loader_name}] Scene penetration B: {scene_B:.4f}', file=file, flush=True)

    print(f'---> [{motion_loader_name}] Human penetration: {human_penetration_dict[motion_loader_name]:.4f}')
    print(f'---> [{motion_loader_name}] Human penetration: {human_penetration_dict[motion_loader_name]:.4f}', file=file, flush=True)

    file.close()

    return foot_slide_dict, foot_penetration_dict, scene_penetration_dict, human_penetration_dict

if __name__ == '__main__':
    pickle_file_root = os.path.join(get_program_root_path(), "Sitcom-Crafter/HSInter/Results-priorMDM")

    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')

    save_path = './evaluation_long_story-{}.txt'.format(pickle_file_root.split("\\")[-1])
    log_file = os.path.join(os.path.dirname(__file__), save_path)
    evaluate_matching_score(pickle_file_root, log_file)