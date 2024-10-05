import sys
sys.path.append(sys.path[0]+"/../")

from datetime import datetime
from HHInter.utils.metrics import *
from collections import OrderedDict
from HHInter.configs import get_config
from os.path import join as pjoin
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from HHInter.datasets import InterHumanDataset
from HHInter.models import *
import copy
from HHInter.datasets.evaluator_models import InterCLIP
from HHInter.global_path import *
import smplx
from HHInter.custom_visualize import axis_angle_to_rot6d
import pickle

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')

class EvaluationDataset(Dataset):
    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats):
        self.device = device
        self.dataset = dataset
        self.max_length = dataset.max_gt_length - 1  # 299, because when transfer to InterGen, we need to calculate adjacent frames.

        # Construct sdf points
        sdf_points_extents = 3.  # InterGen dataset motion maximum extent is 6.3149.
        ceiling_height = 3.
        sdf_points_res = 128

        x = torch.linspace(-sdf_points_extents, sdf_points_extents, sdf_points_res)
        y = torch.linspace(-sdf_points_extents, sdf_points_extents, sdf_points_res)
        z = torch.linspace(-ceiling_height, ceiling_height, sdf_points_res)

        x, y, z = torch.meshgrid(x, y, z)
        self.sdf_coord = torch.stack([x, y, z], dim=-1).permute(3, 0, 1, 2).to(device)

        self.idxs = list(range(len(dataset)))
        random.shuffle(self.idxs)


    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        id = self.idxs[item]
        with torch.no_grad():
            data = self.dataset[id]
            name, text, motion1, motion2, motion_lens, motion_cond_length, motion_R, motion_T, motion_feet, feet_height_thresh, sdf_points = data
            # Collate the data to tensor
            text = [text]
            motion_cond_length = torch.tensor([motion_cond_length]).long()
            motion_R = torch.from_numpy(motion_R).type(torch.float32)
            motion_T = torch.from_numpy(motion_T).type(torch.float32)
            motion_feet = torch.tensor([motion_feet]).long()
            feet_height_thresh = torch.from_numpy(feet_height_thresh).type(torch.float32)
            # Transfer sdf_points to cuda, because 3D data takes much time in the device transition. (~0.5s, but also renders 512MB for 16*4*128*128*128 cuda tensor)
            sdf_points = sdf_points.type(torch.float32).unsqueeze(0).to(self.device)

            sdf_points = torch.cat([self.sdf_coord.unsqueeze(0).expand(len(text), -1, -1, -1, -1),
                                             sdf_points.type(torch.float32)], dim=1)


            # read pickle file based on name.
            with open(pjoin(pickle_file_root, name + '.pkl'), 'rb') as f:
                pred_params = pickle.load(f)

            "============ turn to smplx marker points ==============="
            smplx_model = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplx',
                                            gender='neutral', ext='pkl',
                                            num_betas=10,
                                            num_pca_comps=12,
                                            batch_size=len(pred_params[0])
                                            )
            '''get markers'''
            with open(get_SSM_SMPLX_body_marker_path()) as f:
                markerdict = json.load(f)['markersets'][0]['indices']
            markers = list(markerdict.values())

            motions_output = []
            for iddd in [0,1]:
                bparam = {}
                bparam['transl'] = pred_params[iddd][:, :3]
                bparam['global_orient'] = pred_params[iddd][:, 3:6]
                bparam['body_pose'] = pred_params[iddd][:, 6:69]
                bparam['betas'] = pred_params[iddd][:, 93:103]
                vertices = smplx_model(return_verts=True, **bparam).vertices
                markers_tmp = vertices[:, markers].view(1, len(pred_params[iddd]), -1, 3)
                motions_output.append(markers_tmp)

            motion1_marker, motion2_marker = motions_output[0].cpu(), motions_output[1].cpu()
            B, T = motion1_marker.shape[0], motion1_marker.shape[1]
            motion1_marker = motion1_marker.view(B, T, -1)
            motion2_marker = motion2_marker.view(B, T, -1)
            # +1 here is to align with GT loader output.
            if T < self.max_length + 1:
                padding_len = self.max_length + 1 - T
                D = motion1_marker.shape[-1]
                padding_zeros = torch.zeros((B, padding_len, D))
                motion1_marker = torch.cat((motion1_marker, padding_zeros), dim=1)
                motion2_marker = torch.cat((motion2_marker, padding_zeros), dim=1)
            assert motion1_marker.shape[1] == self.max_length + 1 == motion2_marker.shape[1]

            sub_dict = {'motion_lens': len(motion1_marker[0]),
                        'text': text[0],
                        'sdf_points': sdf_points[0]}

        data = sub_dict
        motion_lens, text, sdf_points = data['motion_lens'], data['text'], data['sdf_points']

        # -1 here because we calculate adjacent frames.
        return "generated", text, motion1_marker[0], motion2_marker[0], motion_lens, motion_cond_length, motion_R, motion_T, motion_feet, feet_height_thresh, sdf_points.cpu()


def get_dataset_motion_loader(opt, batch_size, sdf_points_res):
    opt = copy.deepcopy(opt)
    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.NAME == 'interhuman':
        print('Loading dataset %s ...' % opt.NAME)

        dataset = InterHumanDataset(opt, sdf_points_res, is_eval=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset


def get_motion_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats):
    # Currently the configurations of two datasets are almost the same
    dataset = EvaluationDataset(model, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)

    print('Generated Dataset Loading Completed!!!')

    return motion_loader


def build_models(cfg):
    model = InterCLIP(cfg)

    checkpoint = torch.load(pjoin(os.path.dirname(__file__), '../eval_model/interclip.ckpt'),map_location="cpu")
    # checkpoint = torch.load(pjoin('checkpoints/interclip/model/5.ckpt'),map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return model


def evaluate_matching_score(motion_loaders, file):
    foot_slide_dict = OrderedDict({})
    foot_penetration_dict = OrderedDict({})
    scene_penetration_dict = OrderedDict({})
    human_penetration_dict = OrderedDict({})

    physics_metric = CalculatePhysics()

    # print(motion_loaders.keys())
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        foot_slide = 0
        foot_penetration = 0
        scene_penetration = 0
        human_penetration = 0
        scene_A = 0
        scene_B = 0
        all_size = len(motion_loader.dataset)
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(motion_loader)):
                batch_marker_format = batch
                slide, f_pene = physics_metric.calculate_foot_physics(batch_marker_format, is_gt='ground truth' in motion_loader_name)
                s_pene_A, s_pene_B = physics_metric.calculate_scene_penet(batch_marker_format, is_gt='ground truth' in motion_loader_name)
                h_pene = physics_metric.calculate_human_physics(batch_marker_format, is_gt='ground truth' in motion_loader_name)
                foot_slide += slide
                foot_penetration += f_pene
                scene_penetration += s_pene_A + s_pene_B
                scene_A += s_pene_A
                scene_B += s_pene_B
                human_penetration += h_pene

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

    return foot_slide_dict, foot_penetration_dict, scene_penetration_dict, human_penetration_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Foot sliding': OrderedDict({}),
                                   "Foot penetration": OrderedDict({}),
                                   "Scene penetration": OrderedDict({}),
                                   "Human penetration": OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            foot_slide_dict, foot_penetration_dict, \
                scene_penetration_dict, human_penetration_dict = evaluate_matching_score(motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in foot_slide_dict.items():
                if key not in all_metrics['Foot sliding']:
                    all_metrics['Foot sliding'][key] = [item]
                else:
                    all_metrics['Foot sliding'][key] += [item]

            for key, item in foot_penetration_dict.items():
                if key not in all_metrics['Foot penetration']:
                    all_metrics['Foot penetration'][key] = [item]
                else:
                    all_metrics['Foot penetration'][key] += [item]

            for key, item in scene_penetration_dict.items():
                if key not in all_metrics['Scene penetration']:
                    all_metrics['Scene penetration'][key] = [item]
                else:
                    all_metrics['Scene penetration'][key] += [item]

            for key, item in human_penetration_dict.items():
                if key not in all_metrics['Human penetration']:
                    all_metrics['Human penetration'][key] = [item]
                else:
                    all_metrics['Human penetration'][key] += [item]

        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)

import random
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


if __name__ == '__main__':
    seed_torch(22222)

    priorMDM_flag = True

    if priorMDM_flag:
        pickle_file_root = os.path.join(get_program_root_path(), "priorMDM/results_physical")
    else:
        pickle_file_root = os.path.join(get_program_root_path(), "InterGen/results_physical")

    mm_num_samples = 100
    mm_num_repeats = 30
    mm_num_times = 10

    diversity_times = 300
    replication_times = 3

    # batch_size is fixed to 96!!
    batch_size = 96

    data_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/datasets.yaml")).interhuman_test
    cfg_path_list = [os.path.join(os.path.dirname(__file__), "configs/model.yaml")]


    eval_motion_loaders = {}
    for cfg_path in cfg_path_list:
        model_cfg = get_config(cfg_path)
        eval_motion_loaders[model_cfg.NAME] = lambda: get_motion_loader(
                                                batch_size,
                                                None,
                                                gt_dataset,
                                                device,
                                                mm_num_samples,
                                                mm_num_repeats
                                                )

    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size, sdf_points_res=model_cfg.SDF_POINTS_RES)
    evalmodel_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/eval_model.yaml"))

    save_path = f'./evaluation_priorMDM.log' if priorMDM_flag else f'./evaluation_InterGen.log'
    log_file = os.path.join(os.path.dirname(__file__), save_path)
    evaluation(log_file)