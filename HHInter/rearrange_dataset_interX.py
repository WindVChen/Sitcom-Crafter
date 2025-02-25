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
from scipy.spatial import ConvexHull
from natsort import natsorted, ns
from HHInter.global_path import *
from pytorch3d import transforms


def deep_copy_npz(original_file_path, data_limit):
    # Load original .npz file
    with np.load(original_file_path, allow_pickle=True) as original_data:
        # Create a dictionary to store copied data
        copied_data = {}
        for key in original_data.keys():
            # Deep copy each array
            copied_data[key] = np.copy(original_data[key])
            if key != "gender":
                copied_data[key] = copied_data[key][:data_limit]
    return copied_data


def inv_convert(data, mode='angle'):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data).float()
    else:
        data = data.float()
    rotate_matrix = torch.tensor([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    if mode == 'angle':
        data = transforms.axis_angle_to_matrix(data)
        # convert yup tp zup.
        data = torch.einsum('ij,bjk->bik', rotate_matrix, data)
        # convert to axis angle
        data = transforms.matrix_to_axis_angle(data)
    elif mode == 'trans':
        data = torch.einsum('ij,bj->bi', rotate_matrix, data)
    else:
        raise ValueError
    return data.numpy()

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
    "1) Canonicalize the Inter-X dataset and add in marker information." \
    "2) Downsample fps. " \
    "3) Augment data by reversing the two person."

    #### set input output dataset paths
    interx_data_path = os.path.join(get_dataset_path(), 'Inter-X/motions')
    output_path = os.path.join(get_dataset_path(), 'Inter-X/motions_customized_fps30')
    OUT_FPS = 30

    ## read the corresponding smpx verts indices as markers.
    # Note inter-x is smplx format, different from intergen dataset.
    with open(get_SSM_SMPLX_body_marker_path()) as f:
        marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())

    # There is gender in Inter-X, different from InterGen that are all neutral. If not using gender model, then there will be error like floating person.
    bm_one_neutral = get_body_model('smplx', 'neutral', 1)
    bm_one_female = get_body_model('smplx', 'female', 1)
    bm_one_male = get_body_model('smplx', 'male', 1)

    # To avoid memory error.
    data_limit = 10000
    seg_limit = 5000

    bm_batch_neutral = get_body_model('smplx', 'neutral', seg_limit, device='cuda')
    bm_batch_female = get_body_model('smplx', 'female', seg_limit, device='cuda')
    bm_batch_male = get_body_model('smplx', 'male', seg_limit, device='cuda')

    bodymodel_batch = bm_batch_neutral

    max_spatial_length = np.array([-1, -1, -1]).astype(np.float32)
    block = np.zeros((3, 10))
    block_name = [[[] for j in range(10)] for i in range(3)]
    max_length_name = ""

    seqs_folder = glob.glob(os.path.join(interx_data_path, '*'))

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'person1'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'person2'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'synthetic_scene'), exist_ok=True)

    # random.shuffle(seqs)
    seqs_folder = natsorted(seqs_folder, alg=ns.PATH)
    bar = tqdm(enumerate(seqs_folder), total=len(seqs_folder))
    for idx, seq in bar:
        # if int(os.path.basename(seq)[:-4]) != 142:
        #     continue
        bar.set_description(f"Processing {seq}")
        data_comb_1 = deep_copy_npz(os.path.join(seq, 'P1.npz'), data_limit)
        data_comb_2 = deep_copy_npz(os.path.join(seq, 'P2.npz'), data_limit)
        data_comb = {'person1': data_comb_1, 'person2': data_comb_2}

        fps = 120
        len_subseq = len(data_comb_1['pose_body'])
        assert len_subseq == len(data_comb_2['pose_body'])
        if len_subseq == 0:
            continue

        # Downsample fps
        fps_ratio = float(OUT_FPS) / fps
        new_num_frames = int(fps_ratio * len_subseq)
        downsample_ids = np.linspace(0, len_subseq - 1,
                                     num=new_num_frames, dtype=int)

        # Inter-X dataset is initially Y-up, need to convert to Z-up.
        for p in ['person1', 'person2']:
            data = data_comb[p]
            if data['gender'] == "neutral":
                func = bm_one_neutral
            elif data['gender'] == "female":
                func = bm_one_female
            elif data['gender'] == "male":
                func = bm_one_male
            else:
                raise ValueError

            data['root_orient'] = inv_convert(data['root_orient'])
            data['trans'] = inv_convert(data['trans'], mode='trans')
            # The SMPL rotation is around pelvis, but pelvis is not the original point.
            delta_T = func(betas=torch.from_numpy(data['betas']).repeat(1, 1)).joints[
                      :, 0, :].detach().cpu().numpy()
            data['trans'] = data['trans'] - delta_T + inv_convert(delta_T, mode='trans')

        bparams_record = [{}, {}]
        for order in [['person1', 'person2'], ['person2', 'person1']]:
            transf_rotmat, transf_transl = None, None
            if order[0] == 'person2':
                name = "_swap"
            else:
                name = ""
            for iid, g in enumerate(order):
                data = data_comb[g]

                if data['gender'] == "neutral":
                    func = bm_one_neutral
                    bodymodel_batch = bm_batch_neutral
                elif data['gender'] == "female":
                    func = bm_one_female
                    bodymodel_batch = bm_batch_female
                elif data['gender'] == "male":
                    func = bm_one_male
                    bodymodel_batch = bm_batch_male
                else:
                    raise ValueError

                ## read data
                transl = data['trans']
                pose = np.concatenate(
                    [data['root_orient'].reshape(len_subseq, -1), data['pose_body'].reshape(len_subseq, -1)], axis=-1)
                betas = data['betas'][0]

                # Ensure contact the floor. This is for calculating the foot_contact loss in training.
                segs = len_subseq // seg_limit + (1 if len_subseq % seg_limit != 0 else 0)
                floor_height = 1e10
                for seg in range(segs):
                    start = seg * seg_limit
                    end = min((seg + 1) * seg_limit, len_subseq)
                    if seg != segs - 1:
                        body_param = {}
                        body_param['transl'] = torch.FloatTensor(transl[start:end]).cuda()
                        body_param['global_orient'] = torch.FloatTensor(pose[start:end, :3]).cuda()
                        body_param['betas'] = torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(end - start, 1).cuda()
                        body_param['body_pose'] = torch.FloatTensor(pose[start:end, 3:66]).cuda()
                        smplxout = bodymodel_batch(return_verts=True, **body_param)
                    else:
                        # Need to padding each param's first dimension to meet the seg_limit
                        padding_dimension = seg_limit - (end - start)
                        body_param = {}
                        pad_transl = torch.zeros([padding_dimension, 3], dtype=torch.float32).cuda()
                        pad_global_orient = torch.zeros([padding_dimension, 3], dtype=torch.float32).cuda()
                        pad_betas = torch.zeros([padding_dimension, 10], dtype=torch.float32).cuda()
                        pad_body_pose = torch.zeros([padding_dimension, 63], dtype=torch.float32).cuda()
                        body_param['transl'] = torch.cat([torch.FloatTensor(transl[start:end]).cuda(), pad_transl], dim=0)
                        body_param['global_orient'] = torch.cat([torch.FloatTensor(pose[start:end, :3]).cuda(), pad_global_orient], dim=0)
                        body_param['betas'] = torch.cat([torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(end - start, 1).cuda(), pad_betas], dim=0)
                        body_param['body_pose'] = torch.cat([torch.FloatTensor(pose[start:end, 3:66]).cuda(), pad_body_pose], dim=0)
                        smplxout = bodymodel_batch(return_verts=True, **body_param)
                        smplxout.joints = smplxout.joints[:end - start]

                    "Note: There is case (4861) that one person lie and then stand, and his minimum height of the " \
                    "frame will increase, leading to one person higher than another person. This should be dataset noise, and currently just ignore it."
                    floor_height = min(floor_height, smplxout.joints.detach().squeeze().cpu().numpy()[:, :, 2].min(axis=0).min(axis=0))

                data['trans'] = transl - np.array([0, 0, floor_height])
                transl = data['trans']

                outfilename = os.path.join(output_path, g, os.path.basename(seq) + name + '.npz')

                data_out = {}

                # -==================
                smplxout = None
                for seg in range(segs):
                    start = seg * seg_limit
                    end = min((seg + 1) * seg_limit, len_subseq)
                    if seg != segs - 1:
                        body_param = {}
                        body_param['transl'] = torch.FloatTensor(transl[start:end]).cuda()
                        body_param['global_orient'] = torch.FloatTensor(pose[start:end, :3]).cuda()
                        body_param['betas'] = torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(end - start, 1).cuda()
                        body_param['body_pose'] = torch.FloatTensor(pose[start:end, 3:66]).cuda()
                        smplxout_tmp = bodymodel_batch(return_verts=True, **body_param)
                    else:
                        # Need to padding each param's first dimension to meet the seg_limit
                        padding_dimension = seg_limit - (end - start)
                        body_param = {}
                        pad_transl = torch.zeros([padding_dimension, 3], dtype=torch.float32).cuda()
                        pad_global_orient = torch.zeros([padding_dimension, 3], dtype=torch.float32).cuda()
                        pad_betas = torch.zeros([padding_dimension, 10], dtype=torch.float32).cuda()
                        pad_body_pose = torch.zeros([padding_dimension, 63], dtype=torch.float32).cuda()
                        body_param['transl'] = torch.cat([torch.FloatTensor(transl[start:end]).cuda(), pad_transl], dim=0)
                        body_param['global_orient'] = torch.cat([torch.FloatTensor(pose[start:end, :3]).cuda(), pad_global_orient], dim=0)
                        body_param['betas'] = torch.cat([torch.FloatTensor(betas[:10]).unsqueeze(0).repeat(end - start, 1).cuda(), pad_betas], dim=0)
                        body_param['body_pose'] = torch.cat([torch.FloatTensor(pose[start:end, 3:66]).cuda(), pad_body_pose], dim=0)
                        smplxout_tmp = bodymodel_batch(return_verts=True, **body_param)
                        smplxout_tmp.joints = smplxout_tmp.joints[:end - start]
                        smplxout_tmp.vertices = smplxout_tmp.vertices[:end - start]
                    if smplxout is None:
                        smplxout = smplxout_tmp
                    else:
                        smplxout.joints = torch.cat([smplxout.joints, smplxout_tmp.joints], dim=0)
                        smplxout.vertices = torch.cat([smplxout.vertices, smplxout_tmp.vertices], dim=0)

                ## perform transformation from the world coordinate to the amass coordinate
                ### get transformation from amass space to world space
                if transf_rotmat is None or transf_transl is None:
                    transf_rotmat, transf_transl = get_new_coordinate(smplxout)

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

                bparams_record[iid] = smplxout.vertices.detach().squeeze().cpu().numpy()

                ### extract joints and markers
                joints = smplxout.joints[:, :22, :].detach().squeeze().cpu().numpy()
                markers_67 = smplxout.vertices[:, marker_ssm_67, :].detach().squeeze().cpu().numpy()
                data_out['joints'] = joints
                data_out['marker_ssm2_67'] = markers_67

                for k, v in data_out.items():
                    if k in ['trans', 'poses', 'joints', 'marker_ssm2_67', 'transf_rotmat', 'transf_transl']:
                        data_out[k] = v[downsample_ids]

                np.savez(outfilename, **data_out)

        "Record information (vertex and face of projected convex hull) for synthetic scene construction for each pair of person."
        outfilename_A = os.path.join(output_path, 'synthetic_scene',
                                     os.path.basename(seq).split(".")[0] + name + '.npz')
        outfilename_B = os.path.join(output_path, 'synthetic_scene',
                                     os.path.basename(seq).split(".")[0] + name.replace('_swap', '') + '.npz')
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
