import os.path
import lightning as L
import random

from os.path import join as pjoin

import torch
import smplx

from HHInter.models import *
from collections import OrderedDict
from HHInter.configs import get_config
from HHInter.utils.plot_script import *
from HHInter.utils.preprocess import *

import trimesh
import tqdm
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
import pyrender  # render should be after MeshViewer (which has os.environ=osmesa)
from HHInter.datasets.interhuman import InterHumanDataset
import scipy.ndimage.filters as filters
from HHInter.utils.utils import MotionNormalizerTorch

from HHInter.global_path import *
from HHInter.clip_embedding_extraction import compare_scores

def seed_torch(seed=0):
    print("Seed Fixed!")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class LitGenModel(L.LightningModule):
    def __init__(self, model, cfg, generated_motion_length):
        super().__init__()
        # cfg init
        self.cfg = cfg

        self.generated_motion_length = generated_motion_length
        self.bm = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplx',
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
                                  batch_size=self.generated_motion_length
                                  ).cuda()

        self.automatic_optimization = False

        self.save_root = pjoin(os.path.dirname(__file__), self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # train model init
        self.model = model
        self.normalizer = MotionNormalizerTorch()

        self.body_regressor = MoshRegressor().to('cuda')
        self.body_regressor.load_state_dict(
            torch.load(get_smplx_body_regressor_checkpoint_path(), map_location='cuda')['model_state_dict'])
        self.body_regressor.eval()


    def generate_one_sample(self, batch, name, folder_name, record=False, is_normalize=False):
        self.model.eval()
        motion_output, markers, params = self.generate_script(batch, is_normalize)
        result_path = os.path.join(os.path.dirname(__file__), f"results-{folder_name}/{name}.mp4")
        if not os.path.exists(os.path.dirname(result_path)):
            os.makedirs(os.path.dirname(result_path))

        self.vis_body_pose_beta(motion_output[0], motion_output[1], markers, result_path, record=record)

    @torch.no_grad()
    def generate_script(self, batch, is_normalize, betas=None):
        # Generate the motion_output
        batch = self.model.forward_test(batch)
        motion_output_both = batch["output"].reshape(1, self.generated_motion_length, 2, -1)
        if is_normalize:
            motion_output_both = self.normalizer.backward(motion_output_both)
        motion_output_both = motion_output_both.reshape(1, self.generated_motion_length, 2, 68, 3)
        motion_output_both = motion_output_both[:, :, :, :67, :] + motion_output_both[:, :, :, 67:, :]
        motion_output_both = motion_output_both.reshape(1, self.generated_motion_length, 2, -1)

        # Transform markers to SMPLX parameters
        sequences = [None, None]

        B, T = motion_output_both.shape[:2]

        params = [None, None]

        for j in range(2):
            xb = self.body_regressor(motion_output_both[:, :, j].reshape(-1, 67 * 3), B, T,
                                                             cur_betas=betas[j].to(motion_output_both.device) if betas is not None else None)
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

            x_pred = self.bm(return_verts=True, **body_param)

            sequences[j] = x_pred

            params[j] = xb.cpu().detach().numpy()

        return sequences, motion_output_both.detach().cpu().numpy(), params

    def vis_body_pose_beta(self, body_pose_beta1, body_pose_beta2, markers, save_path, record=False):
        out = FFMpegFileWriter(fps=40.)
        imgs = []
        for fId in tqdm.tqdm(range(body_pose_beta1.vertices.shape[0])):
            body_mesh = trimesh.Trimesh(vertices=body_pose_beta1.vertices[fId].detach().cpu().numpy(),
                                        faces=self.bm.faces, vertex_colors=np.tile(colors['grey'], (10475, 1)))
            body_mesh2 = trimesh.Trimesh(vertices=body_pose_beta2.vertices[fId].detach().cpu().numpy(),
                                         faces=self.bm.faces, vertex_colors=np.tile(colors['pink'], (10475, 1)))

            "Visualize predicted markers and regresserd SMPLX model"
            # points_marker = []
            # for idd in range(67):
            #     tfs = np.eye(4)
            #     tfs[:3, 3] = markers[0][fId][0].reshape(-1, 3)[idd]
            #     sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
            #     sm.visual.vertex_colors = [0.1, 0.9, 0.1, 1.0]
            #     points_marker.append(sm)
            #
            #     tfs = np.eye(4)
            #     tfs[:3, 3] = markers[0][fId][1].reshape(-1, 3)[idd]
            #     sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
            #     sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            #     points_marker.append(sm)
            # points = trimesh.util.concatenate(points_marker)
            #
            # trimesh.util.concatenate([points, body_mesh, body_mesh2]).show()
            if isinstance(mv.viewer, pyrender.OffscreenRenderer):
                mv.set_static_meshes([body_mesh])
                mv.set_dynamic_meshes([body_mesh2])
                body_image = mv.render(render_wireframe=False)
                imgs.append(body_image)
            else:
                mv.viewer.render_lock.acquire()
                mv.set_static_meshes([body_mesh])
                mv.set_dynamic_meshes([body_mesh2])
                mv.viewer.render_lock.release()
        if isinstance(mv.viewer, pyrender.OffscreenRenderer):
            figure = plt.figure(figsize=(12, 8))
            plt.ion()
            plt.tight_layout()
            with out.saving(figure, save_path, dpi=100):
                for img in imgs:
                    plt.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                                        hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.imshow(img)
                    out.grab_frame()
                    plt.pause(0.001)
                    plt.clf()
            plt.close(figure)
        elif record:
            mv.viewer.save_gif(f"{save_path[:-4]}.gif")


def pipeline_merge(sdf, text, motions, betas, hand_pose_retrieval=False):
    generated_motion_length = 300

    # Get __file__ path.
    model_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/model.yaml"))
    infer_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/infer.yaml"))

    model = InterGen(model_cfg, 1)

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        # print not matched weight
        for k in model.state_dict().keys():
            if k not in ckpt["state_dict"]:
                print("Not match: ", k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Checkpoint state from {model_cfg.CHECKPOINT} loaded!")

    litmodel = LitGenModel(model, infer_cfg, generated_motion_length).cuda()
    litmodel.model.eval()

    # Initialize the motion_input
    batch = OrderedDict({})

    B, T = 1, generated_motion_length

    batch["text"] = [text]
    # For marker condition
    batch["motions"] = motions.type(torch.float32).cuda()
    batch["motion_lens"] = torch.tensor([generated_motion_length]).long().cuda()
    batch["motion_cond_length"] = torch.tensor([1]).long().cuda()
    batch["sdf_points"] = sdf.type(torch.float32).cuda() if sdf is not None else None

    motion_output, markers, params = litmodel.generate_script(batch, is_normalize=True, betas=betas)

    if hand_pose_retrieval:
        "Retrieve hand parameters from Inter-X dataset."
        _, _, _, hand_params_1, hand_params_2 = compare_scores(text)
        hand_params_1 = hand_params_1.reshape((len(hand_params_1), -1))
        hand_params_2 = hand_params_2.reshape((len(hand_params_2), -1))

        if len(hand_params_1) > generated_motion_length:
            # randomly sample a sub-segment.
            len_subseq = generated_motion_length
            start_frame = random.randint(0, len(hand_params_1) - len_subseq)
            hand_params_1 = hand_params_1[start_frame:start_frame + len_subseq]
            hand_params_2 = hand_params_2[start_frame:start_frame + len_subseq]
        else:
            # Upsample the segment and ensure the final sequence length equal to generated_motion_length.
            ratio = float(generated_motion_length) / len(hand_params_1)
            new_num_frames = int(ratio * len(hand_params_1))
            upsample_ids = np.linspace(0, len(hand_params_1) - 1,
                                         num=new_num_frames, dtype=int)
            hand_params_1 = hand_params_1[upsample_ids]
            hand_params_2 = hand_params_2[upsample_ids]

            # if not equal, padding with last frame.
            if len(hand_params_1) < generated_motion_length:
                hand_params_1 = np.concatenate([hand_params_1, np.tile(hand_params_1[-1], (generated_motion_length - len(hand_params_1), 1))], axis=0)
                hand_params_2 = np.concatenate([hand_params_2, np.tile(hand_params_2[-1], (generated_motion_length - len(hand_params_2), 1))], axis=0)

        return params, hand_params_1, hand_params_2
    else:
        return params, None, None


if __name__ == '__main__':
    seed_torch(0)
    record = False
    test_dataset = True
    generated_motion_length = 300
    mv = MeshViewer(use_offscreen=False, record=record, zup=True)

    model_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/model.yaml"))
    infer_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/infer.yaml"))

    model = InterGen(model_cfg, 1)

    folder_name = model_cfg.CHECKPOINT.split("/")[-3]

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        # print not matched weight
        for k in model.state_dict().keys():
            if k not in ckpt["state_dict"]:
                print("Not match: ", k)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"Checkpoint state from {model_cfg.CHECKPOINT} loaded!")

    litmodel = LitGenModel(model, infer_cfg, generated_motion_length).cuda()

    if not test_dataset:
        with open(os.path.join(os.path.dirname(__file__), "./prompts.txt")) as f:
            texts = f.readlines()
        texts = [text.strip("\n") for text in texts]

        for text in texts:
            name = text[:48]

            # Initialize the motion_input
            batch = OrderedDict({})

            B, T = 1, generated_motion_length

            batch["text"] = [text]
            # For marker condition
            batch["motions"] = torch.zeros(B, 1, 67*3*2).type(torch.float32).cuda()
            batch["motion_lens"] = torch.tensor([generated_motion_length]).long().cuda()
            batch["motion_cond_length"] = torch.tensor([1]).long().cuda()
            batch["sdf_points"] = torch.zeros(B, 4, model_cfg.SDF_POINTS_RES, model_cfg.SDF_POINTS_RES,
                                              model_cfg.SDF_POINTS_RES).type(torch.float32).cuda()

            litmodel.generate_one_sample(batch, name, folder_name, record=record, is_normalize=model_cfg.Normalizing)
    else:
        # Use test dataset.
        data_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/datasets.yaml")).interhuman_test
        dataset = InterHumanDataset(data_cfg, model_cfg.SDF_POINTS_RES, is_eval=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            shuffle=False,
        )

        for i, zip_input in tqdm.tqdm(enumerate(dataloader)):
            idname, text, motion1, motion2, motion_lens, motion_cond_length, motion_R, motion_T, motion_feet, feet_height_thresh, sdf_points = zip_input

            motions = torch.cat([motion1, motion2], dim=-1)

            "Visualize the conditional motion1 and motion2 markers and regresserd SMPLX model"
            # B, T = motion1.shape[:2]
            # points_marker = []
            # for idd in range(67):
            #     tfs = np.eye(4)
            #     tfs[:3, 3] = motion1[0][0].reshape(-1, 3)[idd].cpu().numpy()
            #     sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
            #     sm.visual.vertex_colors = [0.1, 0.9, 0.1, 1.0]
            #     points_marker.append(sm)
            #
            #     tfs = np.eye(4)
            #     tfs[:3, 3] = motion2[0][0].reshape(-1, 3)[idd].cpu().numpy()
            #     sm = trimesh.creation.uv_sphere(radius=0.03, transform=tfs)
            #     sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            #     points_marker.append(sm)
            # points = trimesh.util.concatenate(points_marker)
            #
            # bm = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplx',
            #                   gender='neutral', ext='pkl',
            #                   num_pca_comps=12,
            #                   create_global_orient=True,
            #                   create_body_pose=True,
            #                   create_betas=True,
            #                   create_left_hand_pose=True,
            #                   create_right_hand_pose=True,
            #                   create_expression=True,
            #                   create_jaw_pose=True,
            #                   create_leye_pose=True,
            #                   create_reye_pose=True,
            #                   create_transl=True,
            #                   batch_size=1
            #                   ).cuda()
            #
            # xb = model.decoder.diffusion.body_regressor(motion1[0, 0].reshape(-1, 67 * 3).float().cuda(), B, T)
            # body_param = {}
            # body_param['transl'] = xb[:, :3]
            # body_param['global_orient'] = xb[:, 3:6]
            # body_param['body_pose'] = xb[:, 6:69]
            # # body_param['left_hand_pose'] = xb[:, 69:81]
            # # body_param['right_hand_pose'] = xb[:, 81:93]
            # body_param['betas'] = xb[:, 93:]
            # x_pred = bm(return_verts=True, **body_param)
            #
            # xb = model.decoder.diffusion.body_regressor(motion2[0, 0].reshape(-1, 67 * 3).float().cuda(), B, T)
            # body_param = {}
            # body_param['transl'] = xb[:, :3]
            # body_param['global_orient'] = xb[:, 3:6]
            # body_param['body_pose'] = xb[:, 6:69]
            # # body_param['left_hand_pose'] = xb[:, 69:81]
            # # body_param['right_hand_pose'] = xb[:, 81:93]
            # body_param['betas'] = xb[:, 93:]
            # x_pred2 = bm(return_verts=True, **body_param)
            #
            # x_pred = trimesh.Trimesh(vertices=x_pred.vertices[0].detach().cpu().numpy(),
            #                 faces=bm.faces, vertex_colors=np.tile(colors['grey'], (10475, 1)))
            # x_pred2 = trimesh.Trimesh(vertices=x_pred2.vertices[0].detach().cpu().numpy(),
            #                 faces=bm.faces, vertex_colors=np.tile(colors['pink'], (10475, 1)))
            #
            # trimesh.util.concatenate([x_pred, x_pred2, points]).show()

            "======================"
            # Construct sdf points
            sdf_points_extents = 3.  # InterGen dataset motion maximum extent is 6.3149.
            ceiling_height = 3.
            sdf_points_res = 128

            x = torch.linspace(-sdf_points_extents, sdf_points_extents, sdf_points_res)
            y = torch.linspace(-sdf_points_extents, sdf_points_extents, sdf_points_res)
            z = torch.linspace(-ceiling_height, ceiling_height, sdf_points_res)

            x, y, z = torch.meshgrid(x, y, z)
            sdf_coord = torch.stack([x, y, z], dim=-1).permute(3, 0, 1, 2)

            name = text[0][:48]

            # Initialize the motion_input
            batch = OrderedDict({})
            batch["motion_lens"] = generated_motion_length

            B, T = motion1.shape[:2]

            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1).type(torch.float32).cuda()
            batch["motion_lens"] = torch.tensor(generated_motion_length).repeat(motion_lens.shape).long().cuda()
            batch["motion_cond_length"] = motion_cond_length.long().cuda()
            batch["sdf_points"] = torch.cat([sdf_coord.unsqueeze(0).expand(B, -1, -1, -1, -1),
                                             sdf_points.type(torch.float32)], dim=1).detach().cuda()

            print(idname, ":", text)

            litmodel.generate_one_sample(batch, name+"_"+str(i), folder_name, record=record, is_normalize=model_cfg.Normalizing)

