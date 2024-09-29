"""open primitive.blend, run script render"""
import bpy
import logging
logger = logging.getLogger(__name__)
import mathutils
from mathutils import Vector, Quaternion, Matrix
import pyquaternion as pyquat
import numpy as np
from glob import glob
import pdb
import os
from pathlib import  Path
import sys
import math
import pickle

def aa2quaternion(aa):
    rod = Vector((aa[0], aa[1], aa[2]))
    angle_rad = rod.length
    axis = rod.normalized()
    return Quaternion(axis, angle_rad)

def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

def convert_str_direction_to_vector(direction):
    return {
      "X": np.array([1., 0., 0.], dtype=np.float64),
      "Y": np.array([0., 1., 0.], dtype=np.float64),
      "Z": np.array([0., 0., 1.], dtype=np.float64),
      "-X": np.array([-1., 0., 0.], dtype=np.float64),
      "-Y": np.array([0., -1., 0.], dtype=np.float64),
      "-Z": np.array([0., 0., -1.], dtype=np.float64),
    }[direction.upper()]

def normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x)
    if norm < eps:
        pdb.set_trace()
    return x / norm


import bpy
import bmesh
from bpy_extras.io_utils import ImportHelper, \
    ExportHelper  # ImportHelper/ExportHelper is a helper class, defines filename and invoke() function which calls the file selector.
import pdb
from mathutils import Vector, Quaternion, Matrix
from math import radians
import numpy as np
import os
import pickle
import random
import glob
# from scipy.spatial.transform import Rotation as R

from bpy.props import (BoolProperty, EnumProperty, FloatProperty, PointerProperty, StringProperty)
from bpy.types import (PropertyGroup)

SMPLX_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
    'right_elbow', 'left_wrist', 'right_wrist',
    'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2', 'left_index3',
    'left_middle1', 'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3',
    'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1', 'left_thumb2',
    'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1',
    'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3',
    'right_ring1', 'right_ring2', 'right_ring3', 'right_thumb1', 'right_thumb2', 'right_thumb3'
]  # same to the definition in https://github.com/vchoutas/smplx/blob/master/smplx/joint_names.py

NUM_SMPLX_JOINTS = len(SMPLX_JOINT_NAMES)
NUM_SMPLX_BODYJOINTS = 21
NUM_SMPLX_HANDJOINTS = 15
ROT_NEGATIVE_X = Matrix(np.array([[1.0000000, 0.0000000, 0.0000000],
                                  [0.0000000, 0.0000000, 1.0000000],
                                  [0.0000000, -1.0000000, 0.0000000]])
                        )
ROT_POSITIVE_Y = Matrix(np.array([[-1.0000000, 0.0000000, 0.0000000],
                                  [0.0000000, 1.0000000, 0.0000000],
                                  [0.0000000, 0.0000000, -1.0000000]])
                        )
'''
note
    - rotation in pelvis is in the original smplx coordinate
    - rotation of the armature is in the blender coordinate
'''


def rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = armature.pose.bones[bone_name].rotation_quaternion
    (axis, angle) = quat.to_axis_angle()
    rodrigues = axis
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues

# Remove default cube
if 'Cube' in bpy.data.objects:
    bpy.data.objects['Cube'].select_set(True)
    bpy.ops.object.delete()


def animate_smplx_one_primitive(armature, scene, data):
    # The added vector value is the location of the pelvis joint (this needs to be dynamically modified, refer to Line 267 in interaction_trainer.py)
    transf_transl = Vector((data['transl']  + np.array([[0.0033, -0.3779, 0.0115]])).reshape(3))

    transl = transf_transl
    global_orient = np.array(data['global_orient']).reshape(3)
    body_pose = np.array(data['body_pose']).reshape(63).reshape(NUM_SMPLX_BODYJOINTS, 3)

    # Update body pose
    for index in range(NUM_SMPLX_BODYJOINTS):
        pose_rodrigues = body_pose[index]
        bone_name = SMPLX_JOINT_NAMES[index + 1]  # body pose starts with left_hip
        set_pose_from_rodrigues(armature, bone_name, pose_rodrigues)

    # set global configurations
    ## set global orientation and translation at local coodinate
    if global_orient is not None:
        armature.rotation_mode = 'QUATERNION'
        global_orient_w = aa2quaternion(global_orient).to_matrix()
        armature.rotation_quaternion = global_orient_w.to_quaternion()

    if transl is not None:
        armature.location = transl

    # Activate corrective poseshapes (update pose blend shapes)
    bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')

def animate_smplx(filepath, render_wpath=False, debug=0):
    print()
    print()

    '''create a new collection for the body and the target'''
    collection_name = "motions_{:05d}".format(random.randint(0, 1000))
    collection_motion = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection_motion)
    collection_targets = bpy.data.collections.new(collection_name + '_targets')
    collection_motion.children.link(collection_targets)

    '''read search results'''
    with open(filepath, "rb") as f:
        dataall = pickle.load(f, encoding="latin1")
    print('read files and setup global info...')
    motiondata = dataall['smplx_param']

    '''add a smplx into blender context'''
    gender = 'neutral'
    bpy.data.window_managers['WinMan'].smplx_tool.smplx_gender = gender
    # bpy.data.window_managers['WinMan'].smplx_tool.smplx_texture = 'smplx_texture_m_alb.png' if gender == 'male' else 'smplx_texture_f_alb.png'
    bpy.ops.scene.smplx_add_gender()

    '''set global variables'''
    obj = bpy.context.object
    if obj.type == 'MESH':
        armature = obj.parent
    else:
        armature = obj
        obj = armature.children[0]
        bpy.context.view_layer.objects.active = obj  # mesh needs to be active object for recalculating joint locations
    bpy.ops.object.smplx_set_texture()  # context needs to be mesh

    print('animate character: {}'.format(obj.name))
    collection_motion.objects.link(armature)  # link it with collection
    collection_motion.objects.link(obj)  # link it with collection
    bpy.context.scene.collection.objects.unlink(armature)  # unlink it from master collection
    bpy.context.scene.collection.objects.unlink(obj)  # unlink it from master collection

    '''update the body shape according to beta'''
    betas = np.array(motiondata["betas"]).reshape(-1).tolist()
    bpy.ops.object.mode_set(mode='OBJECT')
    for index, beta in enumerate(betas):
        key_block_name = f"Shape{index:03}"
        if key_block_name in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks[key_block_name].value = beta
        else:
            print(f"ERROR: No key block for: {key_block_name}")
    ## Update joint locations. This is necessary in this add-on when applying body shape.
    bpy.ops.object.smplx_update_joint_locations('EXEC_DEFAULT')
    print('|-- shape updated...')
    bpy.ops.object.smplx_set_texture()  # context needs to be mesh

    '''move the origin to the body pelvis, and rotate around x by -90degree'''
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')
    deltaT = armature.pose.bones['pelvis'].head.z  # the position at pelvis
    deltax = armature.pose.bones['pelvis'].head.x  # the position at pelvis
    deltay = armature.pose.bones['pelvis'].head.y  # the position at pelvis
    bpy.ops.object.mode_set(mode='POSE')
    # To ensure that the pelvis bone's location is identical to the root one's (note the order and sign)
    armature.pose.bones['pelvis'].location.y -= deltaT
    armature.pose.bones['pelvis'].location.x -= deltax
    armature.pose.bones['pelvis'].location.z += deltay
    armature.pose.bones['pelvis'].rotation_quaternion = ROT_NEGATIVE_X.to_quaternion()
    
    bpy.ops.object.mode_set(mode='OBJECT')

    scene = bpy.data.scenes['Scene']
    '''main loop to update body pose and insert keyframes'''
    animate_smplx_one_primitive(armature, scene, motiondata)

file_path = r'D:\Motion\Story-HIM\HSInter\results\coins\two_stage\test_room\test\optimization_after_get_body\sit on\sit on_sofa_0\0.pkl'
animate_smplx(filepath=file_path, render_wpath=False, debug=0)