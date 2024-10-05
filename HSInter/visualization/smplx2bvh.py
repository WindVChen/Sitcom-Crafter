import sys
sys.path.append(sys.path[0]+"/../")
sys.path.append(sys.path[0]+"/../../")
import torch
import numpy as np
import argparse
import pickle
import smplx
import os

from visualization.utils import bvh, quat
from HHInter.global_path import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=get_SMPL_SMPLH_SMPLX_body_model_path())
    parser.add_argument("--model_type", type=str, default="smplx", choices=["smpl", "smplx"])
    parser.add_argument("--gender", type=str, default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--num_betas", type=int, default=10, choices=[10, 300])
    parser.add_argument("--poses", type=str, default=os.path.join(get_program_root_path(), "Sitcom-Crafter/HSInter/results_Story_HIM_apartment_1/smplh-bvh"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=str, default=os.path.join(get_program_root_path(), "Sitcom-Crafter/HSInter/results_Story_HIM/smplx-bvh"))
    return parser.parse_args()

def smplx2bvh(model_path:str, poses:str, output:str,
             model_type="smpl", gender="MALE", fps=60) -> None:
    """Save bvh file created by smplx parameters.

    Args:
        model_path (str): Path to smplx models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): Defaults to "smplx".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPLX. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """

    names = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "jaw",
        "left_eye_smplhf",
        "right_eye_smplhf",
        "left_index1",
        "left_index2",
        "left_index3",
        "left_middle1",
        "left_middle2",
        "left_middle3",
        "left_pinky1",
        "left_pinky2",
        "left_pinky3",
        "left_ring1",
        "left_ring2",
        "left_ring3",
        "left_thumb1",
        "left_thumb2",
        "left_thumb3",
        "right_index1",
        "right_index2",
        "right_index3",
        "right_middle1",
        "right_middle2",
        "right_middle3",
        "right_pinky1",
        "right_pinky2",
        "right_pinky3",
        "right_ring1",
        "right_ring2",
        "right_ring3",
        "right_thumb1",
        "right_thumb2",
        "right_thumb3"
    ]

    model = smplx.create(model_path=model_path,
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1,
                         use_pca=False)
    
    parents = model.parents.detach().cpu().numpy()

    # Pose setting.
    with open(poses, "rb") as f:
        poses = pickle.load(f)

        rots = np.concatenate([poses[:, 3:69], np.zeros((len(poses), 9)), poses[:, 93:-10]], axis=-1).reshape((-1, 55, 3))
        trans = poses[:, :3]  # (N, 3)
        betas = torch.from_numpy(poses[:, -10:]).type(torch.float32)  # (N, 10)

    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        betas = betas[[0]]
    )
    rest_pose = rest.joints[:, :55].detach().cpu().numpy().squeeze()
    
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 1
    
    # to quaternion
    rots = quat.from_axis_angle(rots)
    
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    # positions[:,0] += trans * 10
    positions[:, 0] += trans
    rotations = np.degrees(quat.to_euler(rots, order=order))
    
    bvh_data ={
        "rotations": rotations[:, :55],
        "positions": positions[:, :55],
        "offsets": offsets[:55],
        "parents": parents[:55],
        "names": names[:55],
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + ".bvh"
    
    bvh.save(output, bvh_data)


if __name__ == "__main__":
    args = parse_args()

    for i in [1, 2]:
        poses = f"{args.poses}/person{i}.pkl"
        output = f"{args.output}/person{i}.bvh"
        smplx2bvh(model_path=args.model_path, model_type=args.model_type,
                 gender=args.gender,
                 poses=poses,
                 fps=args.fps, output=output)
    
    print("finished!")