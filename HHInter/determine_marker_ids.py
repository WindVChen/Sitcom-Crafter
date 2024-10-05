import numpy as np
import pyrender
import trimesh
import smplx
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
import torch
from HHInter.global_path import get_dataset_path, get_program_root_path
import os

SSM_67 = ['C7', 'CLAV', 'LANK', 'LFWT', 'LBAK', 'LBCEP', 'LBSH', 'LBUM', 'LBUST', 'LCHEECK', 'LELB', 'LELBIN', 'LFIN',
          'LFRM2', 'LFTHI', 'LFTHIIN', 'LHEE', 'LIWR', 'LKNE', 'LKNI', 'LMT1', 'LMT5', 'LNWST', 'LOWR', 'LBWT',
          'LRSTBEEF', 'LSHO', 'LTHI', 'LTHMB', 'LTIB', 'LTOE', 'MBLLY', 'RANK', 'RFWT', 'RBAK', 'RBCEP', 'RBSH', 'RBUM',
          'RBUSTLO', 'RCHEECK', 'RELB', 'RELBIN', 'RFIN', 'RFRM2', 'RFRM2IN', 'RFTHI', 'RFTHIIN', 'RHEE', 'RKNE',
          'RKNI', 'RMT1', 'RMT5', 'RNWST', 'ROWR', 'RBWT', 'RRSTBEEF', 'RSHO', 'RTHI', 'RTHMB', 'RTIB', 'RTOE', 'STRN',
          'T8', 'LFHD', 'LBHD', 'RFHD', 'RBHD']


def smplh2smplx(vids):
    smplh2smplx = np.load('smplx_fit2_smplh.npz')['smh2smhf']
    if isinstance(vids, int):
        return int(smplh2smplx[vids])
    return [int(smplh2smplx[vid]) for vid in vids]

# From MOSH++ github repo
all_marker_vids = {'smpl': {'ARIEL': 411,
                            'BHEAD': 384,
                            'C7': 3470,
                            'CHIN': 3052,
                            'CLAV': 3171,
                            'FHEAD': 335,
                            'LAEL': 1655,
                            'LANK': 3327,
                            'LAOL': 1736,
                            'LBAK': 1812,
                            'LBCEP': 628,
                            'LBHD': 182,
                            'LBLLY': 1345,
                            'LBSH': 2940,
                            'LBTHI': 988,
                            'LBUM': 3116,
                            'LBUST': 3040,
                            'LBUSTLO': 1426,
                            'LBWT': 3122,
                            'LCHEECK': 239,
                            'LCHST': 595,
                            'LCLAV': 1298,
                            'LCLF': 1103,
                            'LEBHI': 2274,
                            'LEBHM': 2270,
                            'LEBHP': 2193,
                            'LEBHR': 2293,
                            'LEIDIP': 2295,
                            'LELB': 1666,
                            'LELBIN': 1725,
                            'LEMDIP': 2407,
                            'LEPDIP': 2635,
                            'LEPPIP': 2590,
                            'LEPTIP': 2674,
                            'LERDIP': 2518,
                            'LERPIP': 2478,
                            'LERTIP': 2557,
                            'LETMP': 2070,
                            'LETPIPIN': 2713,
                            'LETPIPOUT': 2711,
                            'LFHD': 0,
                            'LFIN': 2174,
                            'LFOOT': 3365,
                            'LFRM': 1568,
                            'LFRM2': 1741,
                            'LFRM2IN': 1953,
                            'LFRMIN': 1728,
                            'LFSH': 1317,
                            'LFTHI': 874,
                            'LFTHIIN': 1368,
                            'LFWT': 857,
                            'LHEE': 3387,
                            'LHEEI': 3432,
                            'LHPS': 2176,
                            'LHTS': 2134,
                            'LIDX1': 2204,
                            'LIDX2': 2283,
                            'LIDX3': 2320,
                            'LIWR': 2112,
                            'LKNE': 1053,
                            'LKNI': 1058,
                            'LMHAND': 2212,
                            'LMID1': 2389,
                            'LMID2': 2406,
                            'LMID3': 2446,
                            'LMT1': 3336,
                            'LMT5': 3346,
                            'LNECK': 298,
                            'LNWST': 1323,
                            'LOWR': 2108,
                            'LPNK1': 2628,
                            'LPNK2': 2634,
                            'LPNK3': 2674,
                            'LPRFWT': 2915,
                            'LRNG1': 2499,
                            'LRNG2': 2517,
                            'LRNG3': 2564,
                            'LRSTBEEF': 3314,
                            'LSCAP': 1252,
                            'LSHN': 1082,
                            'LSHNIN': 1153,
                            'LSHO': 1861,
                            'LSHOUP': 742,
                            'LTHI': 1454,
                            'LTHILO': 850,
                            'LTHM1': 2251,
                            'LTHM2': 2706,
                            'LTHM3': 2730,
                            'LTHM4': 2732,
                            'LTHMB': 2224,
                            'LTIB': 1112,
                            'LTIBIN': 1105,
                            'LTIP': 1100,
                            'LTOE': 3233,
                            'LUPA': 1443,
                            'LUPA2': 1315,
                            'LWPS': 1943,
                            'LWTS': 1922,
                            'MBLLY': 1769,
                            'MBWT': 3022,
                            'MFWT': 3503,
                            'MNECK': 3057,
                            'RAEL': 5087,
                            'RANK': 6728,
                            'RAOL': 5127,
                            'RBAK': 5273,
                            'RBCEP': 4116,
                            'RBHD': 3694,
                            'RBLLY': 4820,
                            'RBSH': 6399,
                            'RBTHI': 4476,
                            'RBUM': 6540,
                            'RBUST': 6488,
                            'RBUSTLO': 4899,
                            'RBWT': 6544,
                            'RCHEECK': 3749,
                            'RCHST': 4085,
                            'RCLAV': 4780,
                            'RCLF': 4589,
                            'RELB': 5135,
                            'RELBIN': 5194,
                            'RFHD': 3512,
                            'RFIN': 5635,
                            'RFOOT': 6765,
                            'RFRM': 5037,
                            'RFRM2': 5210,
                            'RFRM2IN': 5414,
                            'RFRMIN': 5197,
                            'RFSH': 4798,
                            'RFTHI': 4360,
                            'RFTHIIN': 4841,
                            'RFWT': 4343,
                            'RHEE': 6786,
                            'RHEEI': 6832,
                            'RHPS': 5525,
                            'RHTS': 5595,
                            'RIBHI': 5735,
                            'RIBHM': 5731,
                            'RIBHP': 5655,
                            'RIBHR': 5752,
                            'RIDX1': 5722,
                            'RIDX2': 5744,
                            'RIDX3': 5781,
                            'RIIDIP': 5757,
                            'RIIPIP': 5665,
                            'RIMDIP': 5869,
                            'RIMPIP': 5850,
                            'RIPDIP': 6097,
                            'RIPPIP': 6051,
                            'RIRDIP': 5980,
                            'RIRPIP': 5939,
                            'RITMP': 5531,
                            'RITPIPIN': 6174,
                            'RITPIPOUT': 6172,
                            'RITTIP': 6191,
                            'RIWR': 5573,
                            'RKNE': 4538,
                            'RKNI': 4544,
                            'RMHAND': 5674,
                            'RMID1': 5861,
                            'RMID2': 5867,
                            'RMID3': 5907,
                            'RMT1': 6736,
                            'RMT5': 6747,
                            'RNECK': 3810,
                            'RNWST': 4804,
                            'ROWR': 5568,
                            'RPNK1': 6089,
                            'RPNK2': 6095,
                            'RPNK3': 6135,
                            'RPRFWT': 6375,
                            'RRNG1': 5955,
                            'RRNG2': 5978,
                            'RRNG3': 6018,
                            'RRSTBEEF': 6682,
                            'RSCAP': 4735,
                            'RSHN': 4568,
                            'RSHNIN': 4638,
                            'RSHO': 5322,
                            'RSHOUP': 4230,
                            'RTHI': 4927,
                            'RTHILO': 4334,
                            'RTHM1': 5714,
                            'RTHM2': 6168,
                            'RTHM3': 6214,
                            'RTHM4': 6193,
                            'RTHMB': 5686,
                            'RTIB': 4598,
                            'RTIBIN': 4593,
                            'RTIP': 4585,
                            'RTOE': 6633,
                            'RUPA': 4918,
                            'RUPA2': 4794,
                            'RWPS': 5526,
                            'RWTS': 5690,
                            'SACR': 1783,
                            'STRN': 3506,
                            'T10': 3016,
                            'T8': 3508},
                   'smplx': {
                       "CHN1": 8747,
                       "CHN2": 9066,
                       "LEYE1": 1043,
                       "LEYE2": 919,
                       "REYE1": 2383,
                       "REYE2": 2311,
                       "MTH1": 9257,
                       "MTH2": 2813,
                       "MTH3": 8985,
                       "MTH4": 1693,
                       "MTH5": 1709,
                       "MTH6": 1802,
                       "MTH7": 8947,
                       "MTH8": 2905,
                       "RIDX1": 7611,
                       "RIDX2": 7633,
                       "RIDX3": 7667,
                       "RMID1": 7750,
                       "RMID2": 7756,
                       "RMID3": 7781,
                       "RPNK1": 7978,
                       "RPNK2": 7984,
                       "RPNK3": 8001,
                       "RRNG1": 7860,
                       "RRNG2": 7867,
                       "RRNG3": 7884,
                       "RTHM1": 7577,
                       "RTHM2": 7638,
                       "RTHM3": 8053,
                       "RTHM4": 8068,
                       "LIDX1": 4875,
                       "LIDX2": 4897,
                       "LIDX3": 4931,
                       "LMID1": 5014,
                       "LMID2": 5020,
                       "LMID3": 5045,
                       "LPNK1": 5242,
                       "LPNK2": 5250,
                       "LPNK3": 5268,
                       "LRNG1": 5124,
                       "LRNG2": 5131,
                       "LRNG3": 5149,
                       "LTHM1": 4683,
                       "LTHM2": 4902,
                       "LTHM3": 5321,
                       "LTHM4": 5363,
                       "REBRW1": 2178,
                       "REBRW2": 3154,
                       "REBRW4": 2566,
                       "LEBRW1": 673,
                       "LEBRW2": 2135,
                       "LEBRW4": 1429,
                       "RJAW1": 8775,
                       "RJAW4": 8743,
                       "LJAW1": 9030,
                       "LJAW4": 9046,
                       "LJAW6": 8750,
                       "CHIN3": 1863,
                       "CHIN4": 2946,
                       "RCHEEK3": 8823,
                       "RCHEEK4": 3116,
                       "RCHEEK5": 8817,
                       "LCHEEK3": 9179,
                       "LCHEEK4": 2081,
                       "LCHEEK5": 9168,
                       # 'LETPIPOUT': 5321,
                       'LETPIPIN': 5313,
                       'LETMP': 4840,
                       'LEIDIP': 4897,
                       'LEBHI': 4747,
                       'LEMDIP': 5020,
                       'LEBHM': 4828,
                       'LERTIP': 5151,
                       'LERDIP': 5131,
                       'LERPIP': 5114,
                       'LEBHR': 4789,
                       'LEPDIP': 5243,
                       'LEPPIP': 5232,
                       'LEBHP': 4676,
                       'RITPIPOUT': 8057,
                       'RITPIPIN': 8049,
                       'RITMP': 7581,
                       'RIIDIP': 7633,
                       'RIBHI': 7483,
                       'RIMDIP': 7756,
                       'RIBHM': 7564,
                       'RIRDIP': 7867,
                       'RIRPIP': 7850,
                       'RIBHR': 7525,
                       'RIPDIP': 7984,
                       'RIPPIP': 7968,
                       'RIBHP': 7412
                   }
                   }

all_marker_vids['smplh'] = all_marker_vids['smpl']
all_marker_vids['smplx'].update(
    {k: smplh2smplx(v) for k, v in all_marker_vids['smpl'].items() if k not in all_marker_vids['smplx']})

marker_type_labels = {
    'wrist': [
        'RWRA', 'RIWR',
        'RWRB', 'ROWR',
        'LWRA', 'LIWR',
        'LWRB', 'LOWR',
    ],
    'finger_left': [
        "LIDX1", "LIDX2", "LIDX3",
        "LMID1", "LMID2", "LMID3",
        "LPNK1", "LPNK2", "LPNK3",
        "LRNG1", "LRNG2", "LRNG3",
        "LTHM1", "LTHM2", "LTHM3", "LTHM4", "LTHM6",
        'LETPIPOUT', 'LETPIPIN', 'LETMP', 'LEIDIP', 'LEBHI', 'LEMDIP', 'LEPTIP', 'LETTIP',
        'LEBHM', 'LERTIP', 'LERDIP', 'LERPIP', 'LEBHR', 'LEPDIP', 'LEPPIP', 'LEBHP'

    ],
    'finger_right': [
        "RIDX1", "RIDX2", "RIDX3",
        "RMID1", "RMID2", "RMID3",
        "RPNK1", "RPNK2", "RPNK3",
        "RRNG1", "RRNG2", "RRNG3",
        "RTHM1", "RTHM2", "RTHM3", "RTHM4",
        'RITPIPOUT', 'RITPIPIN', 'RITMP', 'RIIDIP', 'RIBHI', 'RIMDIP', 'RITTIP', 'RIIPIP',
        'RIBHM', 'RIRTIP', 'RIRDIP', 'RIRPIP', 'RIBHR', 'RIPDIP', 'RIPPIP', 'RIBHP', 'RIMPIP'
    ],
    'face': [
        "CHN1", "CHN2",
        "LEYE1", "LEYE2",
        "REYE1", "REYE2",
        "MTH1", "MTH2", "MTH3",
        "MTH4", "MTH5", "MTH6",
        "MTH7", "MTH8", "REBRW1",
        "REBRW2", "REBRW4",
        "LEBRW1", "LEBRW2",
        "LEBRW4", "RJAW1",
        "RJAW4", "LJAW1",
        "LJAW4", "LJAW6",
        "CHIN3", "CHIN4", "RCHEEK3",
        "RCHEEK4", "RCHEEK5", "LCHEEK3", "LCHEEK4", "LCHEEK5"
    ]
}

trans_matrix = np.array([[1.0, 0.0, 0.0, 0],
                         [0.0, 0.0, -1.0, 0],
                         [0.0, 1.0, 0.0, 0],
                         [0.0, 0.0, 0.0, 1]])

def main(
        model_folder=os.path.join(get_dataset_path(), "smplx/models"),
        model_type="smplx",
        ext="pkl",
        gender="neutral",
        plot_markers=True
):
    model = smplx.create(
        model_folder,
        model_type=model_type,
        gender=gender,
        ext=ext,
    )

    vposer, _ = load_model(os.path.join(get_program_root_path(), 'Sitcom-Crafter/HSInter/data/models_smplx_v1_1/models/' + '/vposer_v2_0'), model_code=VPoser,
                                remove_words_in_model_weights='vp_model.', disable_grad=True)
    vposer.eval()

    output = model(
        # body_pose=(vposer.decode(torch.FloatTensor(1, 32).normal_()).get('pose_body')).reshape(1, -1),
        return_verts=True,
    )
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    # process=False to avoid creating a new mesh
    tri_mesh = trimesh.Trimesh(
        vertices, model.faces, vertex_colors=vertex_colors, process=False
    )

    print("displaying first pose, exit window to continue processing")
    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    mesh_list = [tri_mesh]

    scene = pyrender.Scene()
    scene.add(mesh)

    # pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.)
    # camera_pose = np.eye(4)
    # camera_pose[:3, 3] = np.array([0, 0.0, 3.0])
    # scene.add(pc, pose=camera_pose, name='pc-camera')

    if plot_markers:
        for i in range(len(SSM_67)):
            if SSM_67[i] in all_marker_vids[model_type]:
                tfs = np.eye(4)
                tfs[:3, 3] = vertices[all_marker_vids[model_type][SSM_67[i]]]

                sm = trimesh.creation.uv_sphere(radius=0.01, transform=tfs)
                sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]

                joints_pcl = pyrender.Mesh.from_trimesh(sm)
                scene.add(joints_pcl)
                mesh_list.append(sm)
            else:
                raise ValueError(f"Marker {SSM_67[i]} not found in {model_type} markers")
    merge = trimesh.util.concatenate(mesh_list)
    merge.export(f"{model_type}_{gender}.ply")

    pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    main(model_type="smplh", gender="female")
