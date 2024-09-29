import smplx
from HHInter.models import *
from HHInter.global_path import *
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
import json

import trimesh

import os
import random
import numpy as np

from scipy.spatial.transform import Rotation

# Body regressor will not act well for Y-Up, but good for Z-Up (the trained format).
glob_rot = Rotation.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).as_rotvec()
glob_rot = torch.tensor(glob_rot, dtype=torch.float32).reshape(1, -1)

test_type = 'smplh'

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

seed_torch()

while 1:
    model_folder = r"D:\Motion\Envs\smplx\models"
    model_type = "smplx"
    ext = "pkl"
    gender = "neutral"
    vis_list = []

    with open(get_SSM_SMPL_body_marker_path()) as f:
        marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())
    with open(get_SSM_SMPLX_body_marker_path()) as f:
        marker_ssm_67_smplx = list(json.load(f)['markersets'][0]['indices'].values())

    model = smplx.create(
        model_folder,
        model_type=model_type,
        gender=gender,
        ext=ext
    )
    model_h = smplx.create(
        model_folder,
        model_type="smplh",
        gender=gender,
        ext=ext
    )

    vposer, _ = load_model('D:/Motion/Story-HIM/HSInter/data/models_smplx_v1_1/models/' + '/vposer_v2_0', model_code=VPoser,
                           remove_words_in_model_weights='vp_model.', disable_grad=True)
    vposer.eval()

    func = model_h if test_type == 'smplh' else model

    output = func(
        global_orient=glob_rot,
        body_pose=(vposer.decode(torch.FloatTensor(1, 32).normal_()).get('pose_body')).reshape(1, -1),
        return_verts=True,
    )

    sm = trimesh.Trimesh(vertices=output.vertices[0].detach().numpy(),
                         faces=model_h.faces if test_type == 'smplh' else model.faces, vertex_colors=[0, 255, 0, 255])
    vis_list.append(sm)

    markers = output.vertices[0][marker_ssm_67 if test_type == 'smplh' else marker_ssm_67_smplx]

    device = 'cpu'
    body_regressor = MoshRegressor().to(device)
    body_regressor.load_state_dict(
        torch.load(get_smplx_body_regressor_checkpoint_path(), map_location=device)['model_state_dict'])
    body_regressor.eval()

    body_regressor_h = MoshRegressor().to(device)
    body_regressor_h.load_state_dict(
        torch.load(get_smplh_body_regressor_checkpoint_path(), map_location=device)['model_state_dict'])
    body_regressor_h.eval()

    xb = body_regressor(markers.reshape(-1, 67 * 3), 1, 1, torch.ones(1, 1)).reshape(1, -1)

    sub_list = []
    body_param = {}
    body_param['transl'] = xb[:, :3]
    body_param['global_orient'] = xb[:, 3:6]
    body_param['body_pose'] = xb[:, 6:69]
    body_param['betas'] = xb[:, 93:]

    x_pred = model(**body_param, return_verts=True)

    sm = trimesh.Trimesh(vertices=x_pred.vertices[0].detach().numpy(), faces=model.faces, vertex_colors=[255, 0, 0, 255])
    vis_list.append(sm)

    x_pred = x_pred.vertices[0][marker_ssm_67_smplx]

    # Calculate the difference.
    diff = x_pred - markers
    print("-> Smplx: ", diff.norm(dim=1).mean())

    xb = body_regressor_h(markers.reshape(-1, 67 * 3), 1, 1, torch.ones(1, 1)).reshape(1, -1)

    sub_list = []
    body_param = {}
    body_param['transl'] = xb[:, :3]
    body_param['global_orient'] = xb[:, 3:6]
    body_param['body_pose'] = xb[:, 6:69]
    body_param['betas'] = xb[:, 93:]

    x_pred_h = model_h(**body_param, return_verts=True)

    sm = trimesh.Trimesh(vertices=x_pred_h.vertices[0].detach().numpy(), faces=model_h.faces, vertex_colors=[0, 0, 255, 255])
    vis_list.append(sm)

    x_pred_h = x_pred_h.vertices[0][marker_ssm_67]

    # Calculate the difference.
    diff_h = x_pred_h - markers
    print("-> Smplh: ", diff_h.norm(dim=1).mean())

    print("difference between predictions: ", (x_pred - x_pred_h).norm(dim=1).mean())

    trimesh.util.concatenate(vis_list).show()
