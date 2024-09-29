import os, glob
import random

import numpy as np
from tqdm import tqdm
import torch
import smplx
from scipy.spatial.transform import Rotation as R
import json
import pickle
import trimesh
import open3d
from scipy.spatial import ConvexHull
from natsort import natsorted, ns
from HHInter.global_path import *


def bilinear_sample_temporal(data, new_time_steps):
    """
    Bilinearly samples temporal data to obtain new time steps.

    Parameters:
    - data: NumPy array with shape (N, D) where N is the original number of time steps, and D is the dimensionality.
    - new_time_steps: Number of desired time steps (M) after bilinear sampling.

    Returns:
    - NumPy array with shape (M, D) representing bilinearly sampled data.
    """

    # Original number of time steps and dimensionality
    N, D = data.shape

    # Calculate the scale factor for bilinear sampling
    scale_factor = (N - 1) / (new_time_steps - 1)

    # Initialize the result array
    result = np.zeros((new_time_steps, D))

    # Bilinear sampling loop
    for i in range(new_time_steps):
        x = i * scale_factor
        x_floor = int(x)
        x_ceil = min(N - 1, x_floor + 1)
        t = x - x_floor

        # Perform linear interpolation along the time dimension
        value_floor = data[x_floor]
        value_ceil = data[x_ceil]
        interpolated_value = (1 - t) * value_floor + t * value_ceil

        result[i] = interpolated_value

    return result


def calc_calibrate_offset(body_mesh_model, betas, transl, pose):
    '''
    The factors to influence this offset is not clear. Maybe it is shape and pose dependent.
    Therefore, we calculate such delta_T for each individual body mesh.
    It takes a batch of body parameters
    input:
        body_params: dict, basically the input to the smplx model
        smplx_model: the model to generate smplx mesh, given body_params
    Output:
        the offset for params transform
    '''
    n_batches = transl.shape[0]
    bodyconfig = {}
    bodyconfig['body_pose'] = torch.FloatTensor(pose[:, 3:]).cuda()
    bodyconfig['betas'] = torch.FloatTensor(betas).unsqueeze(0).repeat(n_batches, 1).cuda()
    bodyconfig['transl'] = torch.zeros([n_batches, 3], dtype=torch.float32).cuda()
    bodyconfig['global_orient'] = torch.zeros([n_batches, 3], dtype=torch.float32).cuda()
    smplx_out = body_mesh_model(return_verts=True, **bodyconfig)
    delta_T = smplx_out.joints[:, 0, :]  # we output all pelvis locations
    delta_T = delta_T.detach().cpu().numpy()  # [t, 3]

    return delta_T


def get_new_coordinate(smplxout):
    '''
    this function produces transform from body local coordinate to the world coordinate.
    it takes only a single frame.
    local coodinate:
        - located at the pelvis
        - x axis: from left hip to the right hip
        - z axis: point up (negative gravity direction)
        - y axis: pointing forward, following right-hand rule
    '''
    joints = smplxout.joints.detach().cpu().numpy()
    x_axis = joints[:, 2, :] - joints[:, 1, :]
    x_axis[:, -1] = 0
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)
    z_axis = np.tile(np.array([0, 0, 1]), (joints.shape[0], 1))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis, axis=1, keepdims=True)
    global_ori_new = np.stack([x_axis, y_axis, z_axis], axis=-1)
    transl_new = joints[:, :1, :]  # put the local origin to pelvis

    return global_ori_new, transl_new


def get_body_model(type, gender, batch_size, device='cpu'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model_path = get_SMPL_SMPLH_SMPLX_body_model_path()
    body_model = smplx.create(body_model_path, model_type=type,
                              gender=gender, ext='pkl',
                              num_betas=10,
                              batch_size=batch_size
                              )
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model


if __name__ == '__main__':
    "This code does these things: " \
    "1) Canonicalize the Intergen dataset and add in marker information." \
    "2) Downsample fps. " \
    "3) Augment data by reversing the two person."

    #### set input output dataset paths
    intergen_data_path = r'D:\Motion\Dataset\InterGen\motions'
    output_path = r'D:\Motion\Dataset\InterGen\motions_customized'
    OUT_FPS = 40

    ## read the corresponding smplh verts indices as markers.
    with open(r'D:\Motion\Story-HIM\HSInter\data\models_smplx_v1_1\models\markers/SSM-smplh.json') as f:
        marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())

    # InterGen dataset has more than 1,000 different betas, but all is for neutral model.
    bm_one_neutral = get_body_model('smplh', 'neutral', 1)

    max_spatial_length = np.array([-1, -1, -1]).astype(np.float32)
    block = np.zeros((3, 10))
    block_name = [[[] for j in range(10)] for i in range(3)]
    max_length_name = ""

    #### main loop to each subset in AMASS

    seqs = glob.glob(os.path.join(intergen_data_path, '*.pkl'))

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'person1'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'person2'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'synthetic_scene'), exist_ok=True)

    # random.shuffle(seqs)
    seqs = natsorted(seqs, alg=ns.PATH)
    bar = tqdm(enumerate(seqs), total=len(seqs))
    for idx, seq in bar:
        # if int(os.path.basename(seq)[:-4]) != 142:
        #     continue
        bar.set_description(f"Processing {seq}")
        with open(seq, 'rb') as f:
            data_comb = pickle.load(f, encoding='latin1')
        fps = data_comb['mocap_framerate']
        len_subseq = data_comb['frames']
        if len_subseq == 0:
            continue

        # Downsample fps
        fps_ratio = float(OUT_FPS) / fps
        new_num_frames = int(fps_ratio * len_subseq)
        downsample_ids = np.linspace(0, len_subseq - 1,
                                     num=new_num_frames, dtype=int)

        bm_batch_neutral = get_body_model('smplh', 'neutral', len_subseq, device='cuda')
        bodymodel_batch = bm_batch_neutral
        bodymodel_one = bm_one_neutral

        bparams_record = [{}, {}]
        for order in [['person1', 'person2'], ['person2', 'person1']]:
            transf_rotmat, transf_transl = None, None
            if order[0] == 'person2':
                name = "_swap"
            else:
                name = ""
            for iid, g in enumerate(order):
                data = data_comb[g]
                ## read data
                transl = data['trans']
                pose = np.concatenate([data['root_orient'], data['pose_body']], axis=-1)
                betas = data['betas']

                # Ensure contact the floor. This is for calculating the foot_contact loss in training.
                body_param = {}
                body_param['transl'] = torch.FloatTensor(transl).cuda()
                body_param['global_orient'] = torch.FloatTensor(pose[:, :3]).cuda()
                body_param['betas'] = torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(len_subseq, 1).cuda()
                body_param['body_pose'] = torch.FloatTensor(pose[:, 3:66]).cuda()
                smplhout = bodymodel_batch(return_verts=True, **body_param)
                ### extract joints and markers
                # markers_67 = smplhout.vertices[:, marker_ssm_67, :].detach().squeeze().cpu().numpy()
                # Use joints but not vertices, because vertices will lead to human fly on the sky.
                "Note: There is case (4861) that one person lie and then stand, and his minimum height of the " \
                "frame will increase, leading to one person higher than another person. This should be dataset noise, and currently just ignore it."
                floor_height = smplhout.joints.detach().squeeze().cpu().numpy()[:, :, 2].min(axis=0).min(axis=0)
                data['trans'] = transl - np.array([0, 0, floor_height])
                transl = data['trans']

                outfilename = os.path.join(output_path, g, os.path.basename(seq).split(".")[0] + name + '.npz')

                data_out = {}

                # -==================

                body_param = {}
                body_param['transl'] = torch.FloatTensor(transl).cuda()
                body_param['global_orient'] = torch.FloatTensor(pose[:, :3]).cuda()
                body_param['betas'] = torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(len_subseq, 1).cuda()
                body_param['body_pose'] = torch.FloatTensor(pose[:, 3:66]).cuda()

                smplhout = bodymodel_batch(return_verts=True, **body_param)

                ## perform transformation from the world coordinate to the amass coordinate
                ### get transformation from amass space to world space
                if transf_rotmat is None or transf_transl is None:
                    transf_rotmat, transf_transl = get_new_coordinate(smplhout)

                data_out['transf_rotmat'] = transf_rotmat
                data_out['transf_transl'] = transf_transl
                data_out['trans'] = transl
                data_out['poses'] = pose
                data_out['betas'] = betas
                data_out['gender'] = data['gender']
                data_out['mocap_framerate'] = OUT_FPS

                ## under this new coordinate, extract the joints/markers' locations
                ## when get generated joints/markers, one can directly transform them back to world coord
                ## note that hand pose is not considered here.


                bparams_record[iid] = smplhout.vertices.detach().squeeze().cpu().numpy()

                ### extract joints and markers
                joints = smplhout.joints[:, :22, :].detach().squeeze().cpu().numpy()
                markers_67 = smplhout.vertices[:, marker_ssm_67, :].detach().squeeze().cpu().numpy()
                data_out['joints'] = joints
                data_out['marker_ssm2_67'] = markers_67

                for k, v in data_out.items():
                    if k in ['trans', 'poses', 'joints', 'marker_ssm2_67', 'transf_rotmat', 'transf_transl']:
                        data_out[k] = v[downsample_ids]

                np.savez(outfilename, **data_out)

        "Record information (vertex and face of projected convex hull) for synthetic scene construction for each pair of person."
        outfilename_A = os.path.join(output_path, 'synthetic_scene', os.path.basename(seq).split(".")[0] + name + '.npz')
        outfilename_B = os.path.join(output_path, 'synthetic_scene', os.path.basename(seq).split(".")[0] + name.replace('_swap', '') + '.npz')
        data_out = {}

        vertices_A = bparams_record[0]
        vertices_B = bparams_record[1]
        all_mesh = []
        for v_A, v_B in zip(vertices_A[::], vertices_B[::]):
            all_mesh.append(trimesh.Trimesh(v_A, bodymodel_batch.faces, process=False))
            all_mesh.append(trimesh.Trimesh(v_B, bodymodel_batch.faces, process=False))
        # smpl_mot = trimesh.util.concatenate(all_mesh)
        all_mesh = trimesh.util.concatenate(all_mesh)
        spatial_length = all_mesh.extents
        # all_mesh.show()

        max_spatial_length[spatial_length > max_spatial_length] = spatial_length[spatial_length > max_spatial_length]
        # Add 1 to the corresponding position in the block according to spatial_length.astype(int)
        block[range(3), spatial_length.astype(int)] += 1
        block_name[0][spatial_length.astype(int)[0]].append(os.path.basename(seq).split(".")[0])
        block_name[1][spatial_length.astype(int)[1]].append(os.path.basename(seq).split(".")[0])
        block_name[2][spatial_length.astype(int)[2]].append(os.path.basename(seq).split(".")[0])

        print("Block: ", block)

        # max_length_name = os.path.basename(seq).split(".")[0]
        # print("Max spatial length: ", max_spatial_length, "\t name: ", max_length_name)

        all_mesh.vertices[:, 2] = 0

        "Simplify the vertices and faces of the mesh, or it will cause speed problem in dataloader."
        last = all_mesh
        # For 3D points (if they are in one plane), we need to apply convex hull several times until it converges.
        # while len(last.convex_hull.vertices) > len(last.convex_hull.convex_hull.vertices):
        #     last = last.convex_hull
        all_mesh_proj = last

        # For further simplification, we need to remove those points too close.
        # simplex = open3d.geometry.TriangleMesh(
        # vertices=open3d.utility.Vector3dVector(all_mesh_proj.vertices.copy()),
        # triangles=open3d.utility.Vector3iVector(all_mesh_proj.faces.copy()),
        # ).simplify_vertex_clustering(0.1)
        # all_mesh_proj = trimesh.Trimesh(vertices=simplex.vertices, faces=simplex.triangles)

        # Project 3D points to 2D to get fewer vertices and faces, as now we only need to consider one facet.
        point_2d = all_mesh_proj.vertices[:, :2]
        hull = ConvexHull(point_2d)

        point_2d_simplified = point_2d[hull.vertices]
        point_to_3d = np.zeros([len(point_2d_simplified), 3])
        point_to_3d[:, :2] = point_2d_simplified

        # construct faces
        faces = []
        mean_point = point_to_3d.mean(axis=0)
        point_to_3d = np.concatenate([point_to_3d, [mean_point]], axis=0)
        for id in range(len(point_to_3d) - 2):
            faces.append([id, id + 1, len(point_to_3d) - 1])
        faces.append([len(point_to_3d) - 2, 0, len(point_to_3d) - 1])

        all_mesh_proj = trimesh.Trimesh(vertices=point_to_3d, faces=faces)

        data_out['faces'] = np.array(all_mesh_proj.faces.tolist())
        data_out['vertices'] = np.array(all_mesh_proj.vertices.tolist()).astype(np.float32)

        print("Scene vertices: ", data_out['vertices'].shape, "Scene faces: ", data_out['faces'].shape)

        # trimesh.util.concatenate([smpl_mot, all_mesh_proj]).show()
        np.savez(outfilename_A, **data_out)
        np.savez(outfilename_B, **data_out)

    print("Max spatial length: ", max_spatial_length)
    print("Block: ", block)
    print("Block name: ", block_name)
    with open(os.path.join(output_path, 'block_info.pkl'), 'wb') as f:
        pickle.dump([max_spatial_length, block, block_name], f)