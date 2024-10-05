import sys
sys.path.append(sys.path[0]+"/../")
import numpy as np
import torch

from datetime import datetime
from HHInter.datasets import get_dataset_motion_loader, get_motion_loader
from HHInter.models import *
from HHInter.utils.metrics import *
from HHInter.datasets import EvaluatorModelWrapper
from collections import OrderedDict
from HHInter.utils.plot_script import *
from HHInter.utils.utils import *
from HHInter.configs import get_config
from os.path import join as pjoin
from tqdm import tqdm

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')

def build_models(cfg):
    if cfg.NAME == "Sitcom-Crafter":
        model = InterGen(cfg, 1)
    return model

def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    foot_slide_dict = OrderedDict({})
    foot_penetration_dict = OrderedDict({})
    scene_penetration_dict = OrderedDict({})
    human_penetration_dict = OrderedDict({})

    physics_metric = CalculatePhysics()

    # print(motion_loaders.keys())
    print('========== Evaluating MM Distance ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        # if 'ground truth' in motion_loader_name:
        #     continue
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        mm_dist_sum = 0
        top_k_count = 0
        foot_slide = 0
        foot_penetration = 0
        scene_penetration = 0
        human_penetration = 0
        scene_A = 0
        scene_B = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(motion_loader)):
                if 'ground truth' in motion_loader_name:
                    batch_joint_format = batch_marker_format = batch
                else:
                    batch_joint_format, batch_marker_format = batch[:len(batch)//2], batch[len(batch)//2:]
                slide, f_pene = physics_metric.calculate_foot_physics(batch_marker_format, is_gt='ground truth' in motion_loader_name)
                s_pene_A, s_pene_B = physics_metric.calculate_scene_penet(batch_marker_format, is_gt='ground truth' in motion_loader_name)
                h_pene = physics_metric.calculate_human_physics(batch_marker_format, is_gt='ground truth' in motion_loader_name)
                foot_slide += slide
                foot_penetration += f_pene
                scene_penetration += s_pene_A + s_pene_B
                scene_A += s_pene_A
                scene_B += s_pene_B
                human_penetration += h_pene

                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch_joint_format, is_gt='ground truth' in motion_loader_name)
                # print(text_embeddings.shape)
                # print(motion_embeddings.shape)
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                # print(dist_mat.shape)
                mm_dist_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                # print(argsmax.shape)

                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            mm_dist = mm_dist_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = mm_dist
            R_precision_dict[motion_loader_name] = R_precision

            foot_slide_dict[motion_loader_name] = foot_slide / (all_size // batch_size)
            foot_penetration_dict[motion_loader_name] = foot_penetration / (all_size // batch_size)
            scene_penetration_dict[motion_loader_name] = scene_penetration / (all_size // batch_size)
            human_penetration_dict[motion_loader_name] = human_penetration / (all_size // batch_size)

            scene_A = scene_A / (all_size // batch_size)
            scene_B = scene_B / (all_size // batch_size)

            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}', file=file, flush=True)

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

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict, foot_slide_dict, foot_penetration_dict, scene_penetration_dict, human_penetration_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(groundtruth_loader)):
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch, is_gt=True)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())
    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    # print(gt_mu)
    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        # print(mu)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
        eval_dict[model_name] = fid
    return eval_dict


def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                batch[2] = batch[2][0]
                batch[3] = batch[3][0]
                batch[4] = batch[4][0]
                motion_embedings = eval_wrapper.get_motion_embeddings(batch, is_gt=False)
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'MM Distance': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({}),
                                   'Foot sliding': OrderedDict({}),
                                   "Foot penetration": OrderedDict({}),
                                   "Scene penetration": OrderedDict({}),
                                   "Human penetration": OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict, foot_slide_dict, foot_penetration_dict, \
                scene_penetration_dict, human_penetration_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['MM Distance']:
                    all_metrics['MM Distance'][key] = [item]
                else:
                    all_metrics['MM Distance'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

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

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]


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


if __name__ == '__main__':
    mm_num_samples = 100
    mm_num_repeats = 30
    mm_num_times = 10

    diversity_times = 300
    replication_times = 20

    # batch_size is fixed to 96!!
    batch_size = 96

    data_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/datasets.yaml")).interhuman_test
    cfg_path_list = [os.path.join(os.path.dirname(__file__), "configs/model.yaml")]


    eval_motion_loaders = {}
    for cfg_path in cfg_path_list:
        model_cfg = get_config(cfg_path)
        device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(0)
        model = build_models(model_cfg)
        checkpoint = torch.load(model_cfg.CHECKPOINT, map_location=torch.device("cpu"))
        for k in list(checkpoint["state_dict"].keys()):
            if "model" in k:
                checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
        for k in model.state_dict().keys():
            if k not in checkpoint["state_dict"]:
                print("Not match: ", k)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Checkpoint state from {model_cfg.CHECKPOINT} loaded!")

        eval_motion_loaders[model_cfg.NAME] = lambda: get_motion_loader(
                                                batch_size,
                                                model,
                                                gt_dataset,
                                                device,
                                                mm_num_samples,
                                                mm_num_repeats
                                                )

    device = torch.device('cuda:%d' % 0 if torch.cuda.is_available() else 'cpu')
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size, sdf_points_res=model_cfg.SDF_POINTS_RES)
    evalmodel_cfg = get_config(os.path.join(os.path.dirname(__file__), "configs/eval_model.yaml"))
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)

    log_file = os.path.join(os.path.dirname(__file__), f'./evaluation_{model_cfg.CHECKPOINT.split("/")[-3]}.log')
    evaluation(log_file)
