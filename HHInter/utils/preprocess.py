import numpy as np
from HHInter.utils.utils import *

FPS = 40  # The original InterGen model use FPS 30.


def deep_copy_npz(original_file_path):
    # Load original .npz file
    with np.load(original_file_path) as original_data:
        # Create a dictionary to store copied data
        copied_data = {}
        for key in original_data.keys():
            # Deep copy each array
            copied_data[key] = np.copy(original_data[key])
    return copied_data


def load_motion(file_path, min_length):
    try:
        motion = deep_copy_npz(file_path)
    except:
        print("error: ", file_path)
        return None

    if motion['trans'].shape[0] < min_length:
        return None

    return motion


def load_scene(file_path):
    try:
        motion = deep_copy_npz(file_path)
    except:
        print("error: ", file_path)
        return None

    return motion


def load_motion_id(file_path, min_length, id):


    try:
        motion = np.load(file_path).astype(np.float32)[id]
    except:
        print("error: ", file_path)
        return None, None
    motion1 = motion[:, :22 * 3]
    motion2 = motion[:, 62 * 3:62 * 3 + 21 * 6]
    motion = np.concatenate([motion1, motion2], axis=1)

    if motion.shape[0] < min_length:
        return None, None
    return motion