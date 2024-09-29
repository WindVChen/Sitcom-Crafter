import os

import tqdm

os.environ['API_KEY'] = 'AIzaSyCQc_UseY-HguVvknzL9BQAJfdiN16O67Q'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
import pickle
import sys
sys.path.append(sys.path[0]+r"/../")
sys.path.append(sys.path[0]+r"/../../")
import numpy as np
import torch
import trimesh
import shutil
import copy
import time
import ast

sys.path.append(os.getcwd())

from scipy.spatial.transform import Rotation as R
from test_navmesh import *
from exp_GAMMAPrimitive.utils.environments import *
from exp_GAMMAPrimitive.utils import config_env
from pathlib import Path
import collections
import subprocess
from get_scene import ReplicaScene
from scipy.spatial.transform import Rotation
from vis_gen import rollout_primitives
from HHInter.rearrange_dataset import get_new_coordinate
from HHInter.infer import pipeline_merge
from HHInter.utils.slerp_alignment import slerp_poses, slerp_translation, aligining_bodies
from HHInter.global_path import *
from HHInter.models.blocks import MoshRegressor
import google.generativeai as genai
import bisect
from operator import itemgetter
from itertools import groupby
from HHInter.models.losses import GeneralContactLoss

np.random.seed(2333)
torch.manual_seed(2333)

bm_path = get_SMPL_SMPLH_SMPLX_body_model_path()
bm = smplx.create(bm_path, model_type='smplx',
                  gender='neutral', ext='npz',
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
                  batch_size=1
                  ).eval().cuda()


def params2torch(params, dtype=torch.float32):
    return {k: torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v for k, v in params.items()}


def project_to_navmesh(navmesh, points):
    closest, _, _ = trimesh.proximity.closest_point(navmesh, points)
    return closest


def params2numpy(params):
    return {k: v.detach().cpu().numpy() if type(v) == torch.Tensor else v for k, v in params.items()}


def get_navmesh(navmesh_path, scene_path, agent_radius, floor_height=0.0, visualize=False):
    if navmesh_path.exists():
        navmesh = trimesh.load(navmesh_path, force='mesh')
    else:
        scene_mesh = trimesh.load(scene_path, force='mesh')
        """assume the scene coords are z-up"""
        scene_mesh.vertices[:, 2] -= floor_height
        scene_mesh.apply_transform(zup_to_shapenet)
        navmesh = create_navmesh(scene_mesh, export_path=navmesh_path, agent_radius=agent_radius, visualize=visualize)
    navmesh.vertices[:, 2] = 0
    return navmesh


# Judge whether the point is within the navmesh region
def is_inside(navmesh, points_2d):
    points_2d = torch.from_numpy(points_2d).cuda().float()  # [P, 1, 2]
    triangles = torch.cuda.FloatTensor(np.stack([navmesh.vertices[navmesh.faces[:, 0], :2],
                                                 navmesh.vertices[navmesh.faces[:, 1], :2],
                                                 navmesh.vertices[navmesh.faces[:, 2], :2]], axis=-1)).permute(0, 2, 1)[
        None, ...]  # [1, F, 3, 2]

    def sign(p1, p2, p3):
        return (p1[:, :, 0] - p3[:, :, 0]) * (p2[:, :, 1] - p3[:, :, 1]) - (p2[:, :, 0] - p3[:, :, 0]) * (
                p1[:, :, 1] - p3[:, :, 1])

    d1 = sign(points_2d, triangles[:, :, 0, :], triangles[:, :, 1, :])
    d2 = sign(points_2d, triangles[:, :, 1, :], triangles[:, :, 2, :])
    d3 = sign(points_2d, triangles[:, :, 2, :], triangles[:, :, 0, :])

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    inside_triangle = ~(has_neg & has_pos)  # [P, F]
    return inside_triangle.any(-1)[0]


def scene_sdf(mesh_path, sdf_path):
    if os.path.exists(sdf_path):
        with open(sdf_path, 'rb') as f:
            object_sdf = pickle.load(f)
        return object_sdf

    mesh = trimesh.load(mesh_path, force='mesh')
    voxel_resolution = 256

    extents = mesh.bounding_box.extents
    extents = np.array([extents[0] + 2, extents[1] + 2, 0.5])
    transform = np.array([[1.0, 0.0, 0.0, 0],
                          [0.0, 1.0, 0.0, 0],
                          [0.0, 0.0, 1.0, -0.25],
                          [0.0, 0.0, 0.0, 1.0],
                          ])
    transform[:2, 3] += mesh.centroid[:2]
    floor_mesh = trimesh.creation.box(extents=extents,
                                      transform=transform,
                                      )
    scene_mesh = mesh + floor_mesh
    # scene_mesh.show()
    scene_extents = extents + np.array([2, 2, 1])
    scene_scale = np.max(scene_extents) * 0.5
    scene_centroid = mesh.bounding_box.centroid
    scene_mesh.vertices -= scene_centroid
    scene_mesh.vertices /= scene_scale
    sign_method = 'normal'
    surface_point_cloud = get_surface_point_cloud(scene_mesh, surface_point_method='sample', scan_count=100,
                                                  scan_resolution=400, sample_point_count=10000000,
                                                  calculate_normals=(sign_method == 'normal'))

    sdf_grid, gradient_grid = surface_point_cloud.get_voxels(voxel_resolution, sign_method == 'depth', sample_count=11,
                                                             pad=False,
                                                             check_result=False, return_gradients=True)

    object_sdf = {
        'grid': sdf_grid * scene_scale,
        'gradient_grid': gradient_grid,
        'dim': voxel_resolution,
        'centroid': scene_centroid,
        'scale': scene_scale,
    }

    sdf_grids = torch.from_numpy(object_sdf['grid'])
    object_sdf['grid'] = sdf_grids.squeeze().unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)  # 1x1xDxDxD
    if 'gradient_grid' in object_sdf:
        gradient_grids = torch.from_numpy(object_sdf['gradient_grid'])
        object_sdf['gradient_grid'] = gradient_grids.permute(3, 0, 1, 2).unsqueeze(0).to(
            dtype=torch.float32)  # 1x3xDxDxD
    object_sdf['centroid'] = torch.tensor(object_sdf['centroid']).reshape(1, 1, 3).to(dtype=torch.float32)

    with open(sdf_path, 'wb') as f:
        pickle.dump(object_sdf, f)

    return object_sdf


def calc_sdf(vertices, sdf_dict):
    sdf_centroid = sdf_dict['centroid']
    sdf_scale = sdf_dict['scale']
    sdf_grids = sdf_dict['grid']

    batch_size, num_vertices, _ = vertices.shape
    vertices = vertices.reshape(1, -1, 3)  # [B, V, 3]
    vertices = (vertices - sdf_centroid) / sdf_scale  # convert to [-1, 1]
    sdf_values = F.grid_sample(sdf_grids,
                               vertices[:, :, [2, 1, 0]].view(1, batch_size * num_vertices, 1, 1, 3),
                               # [2,1,0] permute because of grid_sample assumes different dimension order, see below
                               padding_mode='border',
                               align_corners=True
                               # not sure whether need this: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
                               ).reshape(batch_size, num_vertices)

    return sdf_values

def llm_order_generation(objects):
    try:
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

        genai.configure(api_key=os.environ['API_KEY'])

        model = genai.GenerativeModel(model_name='gemini-1.5-flash', safety_settings=safety_settings)

        print("\nBelow is the input prompt for the model to generate the plot and motions for the actors based on the given objects:")
        print("===================================================================================================================")

        prompts = "Suppose you are a movie director tasked with creating a plot involving the motions of two actors, " \
                "Amy (female) and Jack (male), in a scene. You will be provided with a list of objects present in the scene. " \
                "Based on these objects, you need to design a plot and specify the motions for the actors. The actors can perform " \
                "three types of motions: walking in the scene, interacting with the objects (only supporting sitting and lying), " \
                "and human-to-human interactions (including handshakes, hugs, fighting, etc.).\n" \
                "Your goal is to ensure the designed plot is reasonable and interesting. You need to output both the plot and the " \
                "specific motions for the actors. Also you should ensure that the output motion orders for the actors are strictly aligned to the order rules. Below is an example to illustrate the input and expected output:\n\n" \
                "### Example\n" \
                "**Input:**\n" \
                "Objects: [chair, table, sofa, stool]\n\n" \
                "**Output Plot:**\n" \
                "Plot: 'Jack and Amy are two friends who meet in a cafe. Jack is sitting on the chair, and Amy walks in and then " \
                "sits on the sofa. They talk to each other. After a while, they stand up and shake hands.'\n\n" \
                "**Output Amy Motion Order:**\n" \
                "Motions: [None | sofa | sit | HHI: the two people greet each other by shaking hands.]\n\n" \
                "**Output Jack Motion Order:**\n" \
                "Motions: [chair | sit | HHI: the two people greet each other by shaking hands.]\n\n" \
                "### Motion Order Rules that need to be strictly followed:\n" \
                "- The motion order for each actor is a list.\n" \
                "- The element in the order list can only be one of the following types: None, the name of an object, a human-to-human interaction description with a prefix 'HHI:', and 'sit' or 'lie'. Other elements are prohibited.\n" \
                "- 'None' denotes the actor will be at a random position in the scene.\n" \
                "- The name of any object denotes the actor will be near that object.\n" \
                "- If 'None' and the name of an object are next to each other, the actor will walk from the random position to the object, and vice versa.\n" \
                "- If two objects are next to each other, the actor will walk from the first object to the second.\n" \
                "- If two 'None' entries are next to each other, the actor will walk from the first random position to the second random position.\n" \
                "- 'sit' and 'lie' denotes the interaction with objects. Ensure such order follows the interacted object's name (e.g., 'chair', 'sit'). Note that only 'sit' and 'lie' are allowed. Other descriptions like 'walk through', 'walk to', 'look at', 'turn on', 'pick up' are prohibited.\n" \
                "- For the human-to-human interaction order, prefix the interaction order with 'HHI:' (e.g., 'HHI: the two performers greet each other by shaking hands.'). There should be the same number of HHI orders in both order lists, " \
                "and these HHI orders should be in the same order, with their context being the same in both order lists. " \
                "Note that there shouldn't be any human names in the HHI descriptions, use terms like 'the person', 'the performer', 'the guy', etc. Also, any descrption about interactions with objects are prohibited in the HHI descriptions. Only interactions between two human bodies are allowed." \
                "Also the HHI description cannot involve motions like sitting or lying. Here are several examples of human-human interaction descriptions:\n" \
                "  - 'the two face each other and make an embracing motion with their arms, then move their arms away from the center and lower them to the ground.'\n" \
                "  - 'one approaches the other and gives them a hug. The other rejects the hug and moves away to another location.'\n" \
                "  - 'one person makes a peace sign with their left hand, and the other person wraps an arm around their shoulder, and they take a group photo together.'\n" \
                "  - 'one person stretches both hands over their head, while the other leans to the right and touches the first person's waist with their right hand.'\n" \
                "  - 'one attempts to hit the head of the other with their left fist, and the other responds by using their left fist to defend.'\n\n" \
                f"With the above example and instructions, please design a plot and generate the motions for the actors based on the given objects: [{objects}]."


        print(prompts)

        response = model.generate_content(prompts)

        print("\nBelow is the generated plot and motions for the actors based on the given objects:")
        print("==================================================================================")

        print(response.text)
        return response.text

    except Exception as e:
        print(e)
        print("get caption failed, try again.....")
        return ''


def llm_order_revision(order_1, order_2):
    try:
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]

        genai.configure(api_key=os.environ['API_KEY'])

        model = genai.GenerativeModel(model_name='gemini-1.5-flash', safety_settings=safety_settings)

        print("\nBelow is the input prompt for the model to revise the given two motion order lists:")
        print("===============================================================================")

        prompts = "You need to make sure the given two motion order lists are strictly aligned to the following rules. If not, please adapt it into the right type by removing/adding/modifying some elements.\n" \
                "### Motion Order Rules:\n" \
                "- No single and double quotation marks in the motion order lists.\n" \
                "- Orders are separated by a vertical bar '|'" \
                "- The element in the order list can only be one of the following types: None, the name of an object, a human-to-human interaction description with a prefix 'HHI:', and 'sit' or 'lie'. Other elements are prohibited.\n" \
                "- If not prefixed with 'HHI:', only 'sit' and 'lie' are allowed. Others like 'walk through', 'walk to', 'look at', 'turn on', 'pick up', etc., are prohibited.\n" \
                "- For the human-to-human interaction order prefixed with 'HHI:', there should be the same number of HHI orders in both order lists, " \
                "and these HHI orders should be in the same order, with their context being the same in both order lists. " \
                "Note that there shouldn't be any human names in the HHI descriptions, use terms like 'the person', 'the performer', 'the guy', etc. Also, any descrption about interactions with objects are prohibited in the HHI descriptions. Only interactions between two human bodies are allowed." \
                "Also the HHI description cannot involve motions like sitting or lying. Here are several examples of human-human interaction descriptions:\n" \
                "  'the two face each other and make an embracing motion with their arms, then move their arms away from the center and lower them to the ground.',\n" \
                "  'one approaches the other and gives them a hug. The other rejects the hug and moves away to another location.',\n" \
                "  'one person makes a peace sign with their left hand, and the other person wraps an arm around their shoulder, and they take a group photo together.',\n" \
                "  'one person stretches both hands over their head, while the other leans to the right and touches the first person's waist with their right hand.',\n" \
                "  'one attempts to hit the head of the other with their left fist, and the other responds by using their left fist to defend.'\n\n" \
                "Please check the following two motion order lists and adapt them into the right type.\n\n" \
                "### Input two motion order lists\n" \
                "**Motion Order List 1:**\n" \
                f"Motions: [{order_1}]\n" \
                "**Motion Order List 2:**\n" \
                f"Motions: [{order_2}]\n" \
                "The output should be only the corrected motion order lists. If the input is correct, please output the same order list. No other explanations or reasonings are needed."

        print(prompts)

        response = model.generate_content(prompts)

        print("\nBelow is the revised motion order lists:")
        print("=========================================")

        print(response.text)
        return response.text

    except Exception as e:
        print(e)
        print("get caption failed, try again.....")
        return ''


def first_scene_preparation(json_path, scene_dir, scene_name):
    """Prepare the scene, including the object meshes and the scene mesh."""
    with open(json_path, 'r') as f:
        semantic_statistic = json.load(f)
    accessible_object = collections.defaultdict(list)
    for obj in semantic_statistic['objects']:
        accessible_object[obj["class_name"]].append(obj)
    print("Accessible object: ", accessible_object.keys())

    if 'replica' in str(scene_dir):
        data_folder = Path('data/replica')
        export_ids = []
        # build from original scene
        if not os.path.exists(data_folder / scene_name / 'mesh_floor.ply'):
            scene = ReplicaScene(scene_name, data_folder, build=True, zero_floor=True)
            scene.mesh.export(data_folder / scene_name / 'mesh_floor.ply')
            for obj_id in range(len(semantic_statistic["id_to_label"])):
                # Need to re-export mesh, because initial exported ones are not subtracted by floor height.
                # if not os.path.exists(scene_dir / 'instances' / f'{obj_id}.ply'):
                obj_category = scene.category_names[obj_id]
                instance_mesh = scene.get_mesh_with_accessory(obj_id)
                instance_mesh.export(scene_dir / 'instances' / f'{obj_id}.ply')
    else:
        print("Scene is not from Replica dataset, please check the scene path.")
        exit(0)
    return accessible_object

def orders_revision(input_orders_1, input_orders_2, accessible_object):
    """Revise the orders to ensure it is reasonable and valid."""
    processed_orders = [None, None]
    for iid, input_orders in enumerate([input_orders_1, input_orders_2]):
        id = 0
        while id < len(input_orders):
            cur_order = input_orders[id]
            if cur_order is None:
                if id != 0 and input_orders[id - 1] is not None and (
                        "sit" in input_orders[id - 1] or "lie" in input_orders[id - 1]):
                    print(f"Need to stand up before walking, automatically add 'stand' order.")
                    input_orders.insert(id, "stand")
            elif "HHI" == cur_order[:3]:
                if id == 0 or id == 1:
                    print(f"Need to have walking order before human-human interaction")
                    # Insert two to ensure walking motions.
                    input_orders.insert(id, None)
                    if id == 0:
                        input_orders.insert(id, None)
                elif input_orders[id - 1] is not None and ("sit" in input_orders[id - 1] or "lie" in input_orders[id - 1]):
                    print(f"Need to stand up before walking, automatically add 'stand' order.")
                    input_orders.insert(id, "stand")
            elif "sit" in cur_order or "lie" in cur_order or "stand" in cur_order:
                if id == 0:
                    print(f"Not support '{cur_order}' interact from start, will skip it.")
                    input_orders.pop(id)
                    continue
                elif input_orders[id - 1] is None:
                    print(f"Not support '{cur_order}' interact without object, will skip it.")
                    input_orders.pop(id)
                    continue
                elif "stand" in cur_order:
                    if "sit" not in input_orders[id - 1] and "lie" not in input_orders[id - 1]:
                        print(f"Not support 'stand' before 'sit' and 'lie', will skip it.")
                        input_orders.pop(id)
                        continue
                else:
                    input_orders[id] = "sit on" if "sit" in cur_order else "lie on"
            elif cur_order in accessible_object.keys():
                if id != 0 and input_orders[id - 1] is not None and (
                        "sit" in input_orders[id - 1] or "lie" in input_orders[id - 1]):
                    print(f"Need to stand up before walking, automatically add 'stand' order.")
                    input_orders.insert(id, "stand")
            else:
                print(f"Order {cur_order} is unrecognizable, will skip it.")
                input_orders.pop(id)
                continue
            id += 1

        if len(input_orders) <= 1:
            print("At least two orders are needed for the demo. Replace the missing object with random sample.")
            input_orders = [None] * (3 - len(input_orders)) + input_orders  # Use 3 to avoid skipped order.

        processed_orders[iid] = input_orders

    # At last, ensure the number of HHI orders are the same. Not put it at first because one case: one person sit then HHI,
    # while the other object/None then HHI. This will lead to different HHI prompt due to the added ' The two performers stand up.'.
    hhi_num_1 = len([order for order in input_orders_1 if order is not None and "HHI:" in order[:4]])
    hhi_num_2 = len([order for order in input_orders_2 if order is not None and "HHI:" in order[:4]])
    if hhi_num_1 != hhi_num_2:
        raise ValueError("The number of HHI orders are not the same, please check the input orders.")
    # Then check if the context of the HHI orders are the same. If not, copy the HHI order context from the first order list to the second order list.
    hhi_orders_1 = [id for id, order in enumerate(input_orders_1) if order is not None and "HHI:" in order[:4]]
    hhi_orders_2 = [id for id, order in enumerate(input_orders_2) if order is not None and "HHI:" in order[:4]]
    for id in range(hhi_num_1):
        if input_orders_1[hhi_orders_1[id]] != input_orders_2[hhi_orders_2[id]]:
            print(f"HHI order {id} in the two order lists are different, will copy the context from the first order list to the second order list.")
            input_orders_2[hhi_orders_2[id]] = input_orders_1[hhi_orders_1[id]]
    return processed_orders


def orders_segmentation(processed_orders):
    """Segment the orders into different parts when encountering the 'stand' order or 'HHI' order"""
    segmented_orders = [None, None]
    for iid, input_orders in enumerate(processed_orders):
        # break the orders with the 'stand' order. And save these segments into a list.
        order_segments = []
        seg = []
        for cur_order in input_orders:
            seg.append(cur_order)
            if cur_order is not None and ("stand" in cur_order or "HHI" in cur_order[:3]):
                if len(seg) > 0:
                    order_segments.append(seg)
                    seg = []
        if len(seg) > 0:
            order_segments.append(seg)
        segmented_orders[iid] = order_segments
    return segmented_orders


def walking_path_points_generation(scene_dir, scene_name, scene_path, segmented_orders, accessible_object, navmesh_loose,
                                   extents, centroid, scene_mesh, floor_height=0.0, visualize=False):
    """Generate the walking path points for the orders. Also will generate the target pose if there are object interaction orders."""
    all_wpaths = [[], []]
    append_vars = [[], []]
    for iid, order_segments in enumerate(segmented_orders):
        # order_segments: [[order1, order2, ...], [order1, order2, ...], ...]
        if iid == 1:
            # Calculate the wpaths distance for `0` order_segments. This is for collision revision.
            # Only consider the first seg, otherwise the logics will be too complex.
            all_intermediate_points_A = np.concatenate(all_wpaths[0][0], axis=0)
            # Calculate the cumulated distance of each point to the first point.
            cumulated_distance_A = np.zeros(len(all_intermediate_points_A))
            cumulated_distance_A[1:] = np.linalg.norm(all_intermediate_points_A[1:] - all_intermediate_points_A[:-1], axis=1)
            cumulated_distance_A = np.cumsum(cumulated_distance_A)
        for indx, seg_orders in enumerate(order_segments):
            # seg_orders: [order1, order2, ...]
            # randomly sample points within the area defined above

            marginal = 0
            only_points_return_flag = False
            if indx != 0:
                "This is to ensure the adjacent segments' wpaths are connected."
                if "HHI" in order_segments[indx - 1][-1][:3]:
                    # Will not calculate wpath, leave the calculation in the latter motion generation step.
                    only_points_return_flag = True
                    points = []
                elif "stand" in order_segments[indx - 1][-1]:
                    # the points in the right is the last segment of the walking path, and -2 denotes sit or lie target point.
                    points = [points[-2]]
                    marginal = 1
                else:
                    raise ValueError("The last order of the previous segment should be 'stand' or 'HHI'.")
            else:
                points = []

            var = []
            repeat_times = 0
            while len(points) < len(seg_orders) + marginal:
                i = len(points) - marginal
                if seg_orders[i] is None:
                    # randomly sample a point within the scene area
                    point = np.random.rand(3) * extents - extents / 2 + centroid
                    point[2] = 0
                elif "HHI" in seg_orders[i][:3]:
                    point = None
                    # For two HHI directly adjacent case, need to add a point (another point will from the last seg) to
                    # ensure there is a walking for frame number alignment.
                    # 'copy' here will be used as a dictation of copying last point in the latter step.
                    if len(points) == 0:  # This case, only_points_return_flag is of course True.
                        points.append('copy')
                    if not only_points_return_flag and len(points) == 1:
                        points.append(points[0])
                elif not ("sit" in seg_orders[i] or "lie" in seg_orders[i] or "stand" in seg_orders[i]):
                    ratio = (2 + repeat_times // 20) if seg_orders[i] != "floor" else 1

                    select = np.random.choice(len(accessible_object[seg_orders[i]]))
                    object = accessible_object[seg_orders[i]][select]

                    # Load instance mesh and use trimesh to extract extents and centroid
                    mesh_path = scene_dir / 'instances' / f'{object["id"]}.ply'
                    mesh = trimesh.load(mesh_path, force='mesh')
                    object_extents = mesh.bounding_box.extents
                    object_centroid = mesh.bounding_box.centroid

                    # The following is not accurate, as 'abb' in the json is actually not the world coordinate.
                    # object_extents = np.array(object["oriented_bbox"]["abb"]["sizes"])
                    # object_centroid = np.array(object["oriented_bbox"]["abb"]["center"])

                    # randomly sample a point in a larger cuboid area
                    point = (np.random.rand(3) * object_extents - object_extents / 2) * ratio
                    point += object_centroid
                    point[2] = 0
                    repeat_times += 1
                elif "sit" in seg_orders[i] or "lie" in seg_orders[i]:
                    action_in = seg_orders[i]
                    action_out = action_in.split(' ')[0]
                    obj_name = accessible_object[seg_orders[i - 1]][select]["id"]
                    mesh_path = scene_dir / 'instances' / f'{obj_name}.ply'
                    sdf_path = scene_dir / 'sdf' / f'{obj_name}_sdf_gradient.pkl'

                    "For 'lie' motion, first use 'sit' motion to determine the human body orientation."
                    if "lie" in seg_orders[i]:
                        action_in_tmp = action_in.replace("lie", "sit")
                        command = (
                            f'python synthesize/coins_sample.py --exp_name test --lr_posa 0.01 --max_step_body 100  '
                            f'--weight_penetration 10 --weight_pose 10 --weight_init 0  --weight_contact_semantic 1 '
                            f'--num_sample 1 --num_try 8  --visualize 1 --full_scene 1 '
                            f'--action \"{action_in_tmp}\" --obj_path \"{mesh_path}\" --obj_category \"{obj_name}\" '
                            f'--obj_id 0 --scene_path \"{scene_path}\" --scene_name \"{scene_name}\"')
                        subprocess.run(command)

                        interaction_path_dir = Path(
                            f'results/coins/two_stage/{scene_name}/test/optimization_after_get_body') / action_in_tmp / f'{action_in_tmp}_{obj_name}_0/'
                        interaction_path_list = list(interaction_path_dir.glob('*.pkl'))
                        interaction_path_list = [p for p in interaction_path_list if p.name != 'results.pkl']

                        target_interaction_path = random.choice(interaction_path_list)
                        with open(target_interaction_path, 'rb') as f:
                            target_interaction = pickle.load(f)
                        smplx_params = target_interaction['smplx_param']
                        body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()

                    command = (f'python synthesize/coins_sample.py --exp_name test --lr_posa 0.01 --max_step_body 100  '
                               f'--weight_penetration 10 --weight_pose 10 --weight_init 0  --weight_contact_semantic 1 '
                               f'--num_sample 1 --num_try 8  --visualize 1 --full_scene 1 '
                               f'--action \"{action_in}\" --obj_path \"{mesh_path}\" --obj_category \"{obj_name}\" '
                               f'--obj_id 0 --scene_path \"{scene_path}\" --scene_name \"{scene_name}\"')
                    print(command)
                    subprocess.run(command)

                    interaction_path_dir = Path(
                        f'results/coins/two_stage/{scene_name}/test/optimization_after_get_body') / action_in / f'{action_in}_{obj_name}_0/'
                    interaction_path_list = list(interaction_path_dir.glob('*.pkl'))
                    interaction_path_list = [p for p in interaction_path_list if p.name != 'results.pkl']

                    interaction_name = 'inter_' + str(obj_name) + '_' + action_out
                    target_point_path = Path('results', 'tmp', scene_name, interaction_name, 'target_point.pkl')
                    target_point_path.parent.mkdir(exist_ok=True, parents=True)
                    target_body_path = Path('results', 'tmp', scene_name, interaction_name, 'target_body.pkl')

                    target_interaction_path = random.choice(interaction_path_list)
                    with open(target_interaction_path, 'rb') as f:
                        target_interaction = pickle.load(f)
                    smplx_params = target_interaction['smplx_param']
                    if 'left_hand_pose' in smplx_params:
                        del smplx_params['left_hand_pose']
                    if 'right_hand_pose' in smplx_params:
                        del smplx_params['right_hand_pose']
                    smplx_params['transl'][:, 2] -= floor_height
                    smplx_params['gender'] = 'male'
                    with open(target_body_path, 'wb') as f:
                        pickle.dump(smplx_params, f)

                    smplx_params = params2torch(smplx_params)
                    pelvis = bm(**smplx_params).joints[0, 0, :].detach().cpu().numpy()
                    r = torch.cuda.FloatTensor(1).uniform_() * 0.2 + 1.0
                    # r = 1.0
                    theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi / 3 - torch.pi / 6
                    if "sit" in seg_orders[i]:
                        body_orient = torch.cuda.FloatTensor(smplx_params['global_orient']).squeeze()

                    # Comment: actually is R @ [0, 0, 1], as [0, 0, 1] is the forward direction in the body frame (y-up).
                    forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
                    forward_dir[2] = 0
                    forward_dir = forward_dir / torch.norm(forward_dir)
                    random_rot = pytorch3d.transforms.euler_angles_to_matrix(torch.cuda.FloatTensor([0, 0, theta]),
                                                                             convention="XYZ")
                    forward_dir = torch.matmul(random_rot, forward_dir)
                    point = pelvis + (forward_dir * r).detach().cpu().numpy()
                    point[2] = 0
                    if not is_inside(navmesh_loose, point[:2].reshape(1, 1, 2)):
                        point = project_to_navmesh(navmesh_loose, np.array([point]))[0]

                    with open(target_point_path, 'wb') as f:
                        pickle.dump(point, f)

                    # These vars are needed in the following step.
                    var = [action_out, interaction_name, target_point_path, target_body_path, mesh_path, sdf_path]
                else:  # stand
                    point = None

                # judge whether the point is within the navmesh region
                if point is not None and not is_inside(navmesh_loose, point[:2].reshape(1, 1, 2)):
                    continue

                # judge whether the point is too close to Character A's point. Only consider the first seg.
                # Logic here is to use the point's cumulated distance to determine the point of similar time in Character
                # A's path. Then compare if the distance between the two points is less than 0.3, which will easily lead to
                # collision. We skip this process for human-scene interaction, whose resampling cost is expensive.
                if collision_revision_flag:
                    if iid == 1 and indx == 0 and not(seg_orders[i] is not None and ("sit" in seg_orders[i] or "lie" in seg_orders[i])) and point is not None:
                        wpaths = [] if len(points) > 0 else [point.reshape(1, -1)]
                        for i in range(len(points)):
                            start_point = points[i]
                            target_point = points[i + 1] if i + 1 < len(points) else point
                            if target_point is None:
                                continue
                            start_target = np.stack([start_point, target_point])
                            wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=False,
                                              scene_mesh=scene_mesh)
                            if len(wpath) == 0:
                                wpath = np.stack([start_point, target_point])
                            if len(wpath) == 1:
                                wpath = np.concatenate([wpath, wpath + np.random.randn(3) * 0.01], axis=0)

                            wpaths.append(wpath)
                        all_intermediate_points_B = np.concatenate(wpaths, axis=0)
                        if len(all_intermediate_points_B) > 1:
                            # Calculate the cumulated distance of current point to the first point.
                            distance = np.linalg.norm(all_intermediate_points_B[1:] - all_intermediate_points_B[:-1], axis=1).sum()
                        else:
                            distance = 0
                        # bisect determine 'distance' index in 'cumulated_distance_A'.
                        index = bisect.bisect_left(cumulated_distance_A, distance)
                        # judge if point is within 0.3 scope of the point of Character A, if so, continue to resample.
                        if index < len(all_intermediate_points_A):
                            if np.linalg.norm(all_intermediate_points_A[index] - point) < 0.3:
                                continue
                            if index + 1 < len(all_intermediate_points_A):
                                if np.linalg.norm(all_intermediate_points_A[index + 1] - point) < 0.3:
                                    continue
                        else:
                            if np.linalg.norm(all_intermediate_points_A[-1] - point) < 0.3:
                                continue

                points.append(point)
                repeat_times = 0

            # Visualize points
            if visualize:
                scene = pyrender.Scene()
                scene.add(pyrender.Mesh.from_trimesh(scene_mesh))
                navmesh_loose_copy = deepcopy(navmesh_loose)
                navmesh_loose_copy.vertices[:, 2] += 0.2
                navmesh_loose_copy.visual.vertex_colors = np.array([0, 0, 200, 50])
                scene.add(pyrender.Mesh.from_trimesh(navmesh_loose_copy))
                for p in points:
                    if p is None:
                        continue
                    sm = trimesh.creation.uv_sphere(radius=0.05)
                    sm.visual.vertex_colors = [1.0, 0.0, 0.0]
                    tfs = np.tile(np.eye(4), (1, 1, 1))
                    tfs[:, :3, 3] = p
                    scene.add(pyrender.Mesh.from_trimesh(sm, poses=tfs))
                viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

            if only_points_return_flag:
                all_wpaths[iid].append(points)
            else:
                wpaths = []
                # From the start to the end of the points, iteratively sample two adjacent points as the start and target points
                for i in range(len(points) - 1):
                    start_point = points[i]
                    target_point = points[i + 1]
                    if target_point is None:  # stand
                        continue
                    start_target = np.stack([start_point, target_point])
                    # find collision free path
                    wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=False,
                                      scene_mesh=scene_mesh)

                    # When point is at edges, maybe no path.
                    if len(wpath) == 0:
                        wpath = np.stack([start_point, target_point])
                    # when the last is stand and the current is only HHI order, wpath will be only one (due to copy of stand).
                    if len(wpath) == 1:
                        wpath = np.concatenate([wpath, wpath + np.random.randn(3) * 0.01], axis=0)

                    wpaths.append(wpath)

                all_wpaths[iid].append(wpaths)
            append_vars[iid].append(var)
    return all_wpaths, append_vars


def mid_further_split_orders(all_wpaths, segmented_orders, append_vars):
    """Further splitting mid-order according to HHI text. This is to ensure each segment has the same length and also to suffice
    the HHI interaction generation that behind the two person locomotion/HSI has ended."""
    humaninter_segs = []
    tmp_A = []
    tmp_B = []
    iiid_A = 0
    iiid_B = 0
    while iiid_A < len(all_wpaths[0]):
        wpaths_A, seg_orders_A, var_A = all_wpaths[0][iiid_A], segmented_orders[0][iiid_A], append_vars[0][iiid_A]
        # HHI text will be always at the end of the orders if exists, due to the above segment script.
        if seg_orders_A[-1] is not None and "HHI" in seg_orders_A[-1][:3]:
            tmp_B = []
            while True:
                wpaths_B, seg_orders_B, var_B = all_wpaths[1][iiid_B], segmented_orders[1][iiid_B], append_vars[1][
                    iiid_B]
                iiid_B += 1
                if seg_orders_B[-1] == seg_orders_A[-1]:
                    # Avoid redundant empty segments when two HHInter segments are next to each other.
                    if len(tmp_A) > 0 or len(tmp_B) > 0:
                        humaninter_segs.append([tmp_A, tmp_B])
                    tmp_B = []
                    tmp_B.append([wpaths_B, seg_orders_B, var_B])
                    break
                else:
                    tmp_B.append([wpaths_B, seg_orders_B, var_B])
            tmp_A = []
            tmp_A.append([wpaths_A, seg_orders_A, var_A])
            # Ensure the HHInter orders are independently separated from others.
            humaninter_segs.append([tmp_A, tmp_B])
            tmp_A = tmp_B = []
        else:
            tmp_A.append([wpaths_A, seg_orders_A, var_A])
        iiid_A += 1
    if len(tmp_A) > 0 or iiid_B < len(all_wpaths[1]):
        tmp_B = []
        while iiid_B < len(all_wpaths[1]):
            wpaths_B, seg_orders_B, var_B = all_wpaths[1][iiid_B], segmented_orders[1][iiid_B], append_vars[1][iiid_B]
            tmp_B.append([wpaths_B, seg_orders_B, var_B])
            iiid_B += 1
        humaninter_segs.append([tmp_A, tmp_B])
    return humaninter_segs


def motion_generation(humaninter_segs, result_path, scene_path, scene_name, navmesh_tight_path, navmesh_loose,
                      scene_mesh, wpath_path, path_name, scene_sdf_path, floor_height=0.0):
    for ind, humaninter_seg in enumerate(humaninter_segs):
        # humaninter_seg: [[[wpaths_A1, seg_orders_A1, var_A1], [wpaths_A2, seg_orders_A2, var_A2], ...],
        #                   [[wpaths_B1, seg_orders_B1, var_B1], [wpaths_A2, seg_orders_A2, var_A2], ...]]

        if (result_path / 'person1.pkl').exists():
            with open(result_path / 'person1.pkl', 'rb') as f:
                data = pickle.load(f)
                motions = data['motion']
                smplx_param_A = rollout_primitives(motions)
        else:
            smplx_param_A = []

        if (result_path / 'person2.pkl').exists():
            with open(result_path / 'person2.pkl', 'rb') as f:
                data = pickle.load(f)
                motions = data['motion']
                smplx_param_B = rollout_primitives(motions)
        else:
            smplx_param_B = []

        "To ensure motion consistency when the last seg is HHI, which needs to be after HHI generation."
        if ind != 0:
            for iid, smplx_param in zip([0, 1], [smplx_param_A, smplx_param_B]):
                # There will be case that humaninter_seg is A:[], B:[...], thus need to check the last last segment.
                if (len(humaninter_segs[ind - 1][iid]) and humaninter_segs[ind - 1][iid][-1][1][-1][:3] == "HHI") or \
                        (not len(humaninter_segs[ind - 1][iid]) and ind >= 2 and humaninter_segs[ind - 2][iid][-1][1][-1][:3] == "HHI"):
                    "In this case, the return wpath is still sampled points (search 'only_points_return_flag' in the former step)," \
                    "Need to add the final position of the last generated HHI motion to the points and then calculate the wpath."
                    points = ([np.append(smplx_param[-1][:2], 0)] + humaninter_seg[iid][0][0]) if len(smplx_param) > 0 and len(humaninter_seg[iid]) else []

                    wpaths = []
                    # From the start to the end of the points, iteratively sample two adjacent points as the start and target points
                    for i in range(len(points) - 1):
                        start_point = points[i]
                        target_point = points[i + 1]
                        if target_point == 'copy':
                            target_point = points[i]
                        if target_point is None:  # stand or HHI
                            continue
                        start_target = np.stack([start_point, target_point])
                        # find collision free path
                        wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=False,
                                          scene_mesh=scene_mesh)

                        # When point is at edges, maybe no path.
                        if len(wpath) == 0:
                            wpath = np.stack([start_point, target_point])
                        # when use 'copy', only return one point.
                        if len(wpath) == 1:
                            wpath = np.concatenate([wpath, wpath + np.random.randn(3) * 0.01], axis=0)

                        wpaths.append(wpath)
                    if len(humaninter_seg[iid]):
                        humaninter_seg[iid][0][0] = wpaths

        "Ensure the cumulated motion time of the two persons are consistent. Index [1] because the HHInter " \
        "orders are independently separated from others."
        if len(humaninter_seg[0]) and humaninter_seg[0][0][1][-1] is not None and humaninter_seg[0][0][1][-1][:3] == "HHI":
            human_inter_flag = True

            # Wpaths number.
            # +2 because person2 needs to walk to the front of the person1.
            max_frames = max(len(np.concatenate(humaninter_seg[0][0][0], axis=0)) if len(humaninter_seg[0][0][0]) != 0 else 0,
                             (len(np.concatenate(humaninter_seg[1][0][0], axis=0)) + 2) if len(humaninter_seg[1][0][0]) != 0 else 2) * 7

            # Supple the frames to ensure the two persons have the same motion time.
            if len(smplx_param_A) > len(smplx_param_B):
                frames_supple = np.array([0, (len(smplx_param_A) - len(smplx_param_B) + 5) // 10]) + max_frames
            else:
                frames_supple = np.array([(len(smplx_param_B) - len(smplx_param_A) + 5) // 10, 0]) + max_frames
        else:
            human_inter_flag = False

        "----------------------------------------------------------------------------------------"
        "Generate single person motion and human-object interaction"
        "----------------------------------------------------------------------------------------"
        for iid, multipel_vars in enumerate(humaninter_seg):
            if (result_path / f'person{iid + 1}.pkl').exists():
                last_motion_path = result_path / f'person{iid + 1}.pkl'
            else:
                last_motion_path = None

            for iiid, (wpaths, seg_orders, var) in enumerate(multipel_vars):
                print("Current generation order: ", seg_orders)
                if len(var):
                    action_out, interaction_name, target_point_path, target_body_path, mesh_path, sdf_path = var

                # wpaths = np.array([[[0.0,0.0, 0], [0.4, 0.2, 0]], [[0.3,1.3, 0], [0.4, 1.0, 0]]])[[iid]]
                # wpath = wpaths[0]

                if human_inter_flag:
                    "Ensure the forward directions are opposite for the human interaction."
                    if iid == 1:
                        with open(result_path / 'person1.pkl', 'rb') as f:
                            data = pickle.load(f)
                            motions = data['motion']
                            smplx_param_A = rollout_primitives(motions)

                            # Get the forward direction of the first person.
                            body_orient = torch.cuda.FloatTensor(smplx_param_A[-1, 3:6]).squeeze()
                            forward_dir = pytorch3d.transforms.axis_angle_to_matrix(body_orient)[:, 2]
                            forward_dir[2] = 0
                            forward_dir = forward_dir / torch.norm(forward_dir)

                            try_num = 0
                            offset_coeff = 0.3
                            while try_num < 10:
                                r = torch.FloatTensor(1).uniform_() * 0.2 + 1.
                                theta = torch.cuda.FloatTensor(1).uniform_() * torch.pi - torch.pi / 2
                                random_rot = pytorch3d.transforms.euler_angles_to_matrix(
                                    torch.cuda.FloatTensor([0, 0, theta]),
                                    convention="XYZ")
                                first_person_dir = torch.matmul(random_rot, forward_dir).cpu().numpy()

                                "Add two points to walk to the front of the first person and then turn body back."
                                additional_front_point = smplx_param_A[-1, :3] + r.cpu().numpy() * first_person_dir

                                if is_inside(navmesh_loose, additional_front_point[:2].reshape(1, 1, 2)):
                                    break
                                else:
                                    try_num += 1

                            if not is_inside(navmesh_loose, additional_front_point[:2].reshape(1, 1, 2)):
                                additional_front_point = project_to_navmesh(navmesh_loose, np.array([additional_front_point]))[0]
                            additional_front_point[2] = 0

                            additional_point = additional_front_point - offset_coeff * first_person_dir
                            if not is_inside(navmesh_loose, additional_point[:2].reshape(1, 1, 2)):
                                additional_point = project_to_navmesh(navmesh_loose, np.array([additional_point]))[0]
                            additional_point[2] = 0

                            points = [wpaths[-1][-1], additional_front_point, additional_point]

                            # From the start to the end of the points, iteratively sample two adjacent points as the start and target points
                            tmp_wpaths = []
                            for i in range(len(points) - 1):
                                start_point = points[i]
                                target_point = points[i + 1]
                                if target_point is None:  # stand
                                    continue
                                start_target = np.stack([start_point, target_point])
                                # find collision free path
                                wpath = path_find(navmesh_loose, start_target[0], start_target[1], visualize=False,
                                                  scene_mesh=scene_mesh)

                                # When point is at edges, maybe no path.
                                if len(wpath) == 0:
                                    wpath = np.stack([start_point, target_point])

                                tmp_wpaths.append(wpath)

                            wpaths[-1] = np.concatenate(
                                [wpaths[-1], np.concatenate(tmp_wpaths, axis=0)[1:]],
                                axis=0)

                if len(wpaths):  # Decide if pure interaction.
                    wpath = np.concatenate(wpaths, axis=0)

                    with open(wpath_path, 'wb') as f:
                        pickle.dump(wpath, f)

                    # If HHInter, the max_depth is set to frames_supple to ensure the two persons have the same motion time.
                    max_depth = 30 * len(wpath) if not human_inter_flag else frames_supple[iid]

                    cfg_policy = 'MPVAEPolicy_frame_label_walk_collision/map_nostop'
                    cfg_policy_path = f'../results/exp_GAMMAPrimitive/{cfg_policy}'
                    command = (
                        f"python synthesize/gen_locomotion_unify.py --goal_thresh 0.5 --goal_thresh_final 0.25 --max_depth {max_depth} --num_gen1 128 --num_gen2 16 --num_expand 8 "
                        f"--project_dir . --cfg_policy {cfg_policy_path} "
                        f"--gen_name policy_search --num_sequence 1 "
                        f"--scene_path {scene_path} --scene_name {scene_name} --navmesh_path {navmesh_tight_path} --floor_height {floor_height:.2f} --wpath_path {wpath_path} --path_name {path_name} "
                        f"--weight_pene 1 "
                        f"--visualize 0 --use_zero_pose 1 --use_zero_shape 1 --random_orient 0 --clip_far 1"
                    )
                    if last_motion_path is not None:
                        command += f" --last_motion_path {last_motion_path}"

                    # If HHInter, the no_early_stop is set to ensure the two persons have the same motion time.
                    if human_inter_flag:
                        command += f"  --no_early_stop"

                    subprocess.run(command)
                    last_motion_path = f'results/locomotion/{scene_name}/{path_name}/{cfg_policy}/policy_search/seq000/results_ssm2_67_condi_marker_map_0.pkl'

                # If HHInter, there is only single motion before HHInteraction.
                if human_inter_flag:
                    continue

                if seg_orders[-1] is not None and (
                        "sit" in seg_orders[-1] or "lie" in seg_orders[-1] or "stand" in seg_orders[-1]):
                    # TODO: whether need to decide 1/2 frame policy by the existence of pre-locomotion.
                    seq_name = interaction_name + '_down'
                    command = "python synthesize/gen_interaction_unify.py --goal_thresh_final -1 --max_depth 15 --num_gen1 128 --num_gen2 32 --num_expand 4 " \
                              "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_{}_marker/{}_2frame " \
                              "--gen_name policy_search --num_sequence 1 " \
                              "--random_seed 1 --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                              "--target_body_path {} --interaction_name {} --start_point_path {} " \
                              "--use_zero_pose 1 --weight_target_dist 1 --history_mode 2 " \
                              "--visualize 0".format(action_out, action_out, scene_path, scene_name, sdf_path,
                                                     mesh_path,
                                                     floor_height,
                                                     target_body_path,
                                                     seq_name, target_point_path)
                    if last_motion_path is not None:
                        command += f" --last_motion_path {last_motion_path}"
                    print(command)
                    subprocess.run(command)

                    last_motion_path = f'results/interaction/{scene_name}/{seq_name}/MPVAEPolicy_{action_out}_marker/{action_out}_2frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'

                if seg_orders[-1] is not None and ("stand" in seg_orders[-1]):  # stand
                    """stand up"""
                    command = "python synthesize/gen_interaction_unify.py --goal_thresh_final 0.3 --max_depth 10 --num_gen1 128 --num_gen2 16 --num_expand 8 " \
                              "--project_dir . --cfg_policy ../results/exp_GAMMAPrimitive/MPVAEPolicy_{}_marker/{}_2frame " \
                              "--gen_name policy_search --num_sequence 1 " \
                              "--random_seed 1 --scene_path {} --scene_name {} --sdf_path {} --mesh_path {} --floor_height {:.2f} " \
                              "--target_point_path {} --interaction_name {} " \
                              "--use_zero_pose 0 --weight_target_dist 1 --history_mode 2 " \
                              "--visualize 0".format(action_out, action_out, scene_path, scene_name, sdf_path,
                                                     mesh_path,
                                                     floor_height,
                                                     target_point_path, interaction_name + '_up')
                    if last_motion_path is not None:
                        command += f" --last_motion_path {last_motion_path}"
                    print(command)
                    subprocess.run(command)
                    last_motion_path = f'results/interaction/{scene_name}/{interaction_name}_up/MPVAEPolicy_{action_out}_marker/{action_out}_2frame/policy_search/seq000/results_ssm2_67_condi_marker_inter_0.pkl'

            if last_motion_path is not None:
                if last_motion_path != result_path / f'person{iid + 1}.pkl':
                    shutil.copy(last_motion_path, result_path / f'person{iid + 1}.pkl')
        "----------------------------------------------------------------------------------------"

        "----------------------------------------------------------------------------------------"
        "After fininshing the single motion of the two persons, we need to do the HHInteraction."
        "----------------------------------------------------------------------------------------"
        if human_inter_flag:
            text = humaninter_seg[0][0][1][-1][4:]
            with open(get_SSM_SMPLX_body_marker_path()) as f:
                marker_ssm_67 = list(json.load(f)['markersets'][0]['indices'].values())

            "Convert the markers of the last motion of the two persons to the canonical markers."
            with open(result_path / 'person1.pkl', 'rb') as f:
                data = pickle.load(f)
                motions = data['motion']
                smplx_param_A = rollout_primitives(motions)
                smplx_param_A_old = smplx_param_A[-1].copy()

                betas_A = motions[0]['betas']

                smplx_dict = {
                    'betas': betas_A[np.newaxis],
                    'transl': smplx_param_A[-1, :3][np.newaxis],
                    'global_orient': smplx_param_A[-1, 3:6][np.newaxis],
                    'body_pose': smplx_param_A[-1, 6:69][np.newaxis],
                    'return_verts': True
                }
                smplx_dict = params2torch(smplx_dict)
                smplhout = bm(**smplx_dict)
                marker_pos_A = smplhout.vertices[:, marker_ssm_67, :].detach().cpu().numpy()
                joint_pos_A = smplhout.joints.detach().cpu().numpy()
                transf_rotmat, transf_transl = get_new_coordinate(smplhout)

                marker_pos_A = np.einsum('ij,bpj->bpi', transf_rotmat[0].T, marker_pos_A - transf_transl)
                joint_pos_A = np.einsum('ij,bpj->bpi', transf_rotmat[0].T, joint_pos_A - transf_transl)

                # init_A is for copying the generated motion to it.
                init_A = copy.deepcopy(data['motion'][-1])
                init_A['transf_rotmat'] = transf_rotmat
                init_A['transf_transl'] = transf_transl
                init_A['mp_type'] = '0-frame'  # Avoid the first 1/2 frames being removed in vis_gen code.

                "Convert the global orientation and transl of the last motion of the two persons to the canonical ones. " \
                "This conversion is for the following SLERP and alignment."
                delta_T = bm(betas=torch.FloatTensor(motions[0]['betas']).repeat(1, 1).cuda()).joints[
                          :, 0, :].detach().cpu().numpy()
                # get new global_orient
                global_ori = Rotation.from_rotvec(
                    smplx_param_A_old[3:6][np.newaxis]).as_matrix()  # to [t,3,3] rotation mat
                global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat[0].T, global_ori)
                smplx_param_A_old[3:6] = Rotation.from_matrix(global_ori_new).as_rotvec()[0]
                # get new transl
                transl = np.einsum('ij,tj->ti', transf_rotmat[0].T,
                                   smplx_param_A_old[:3][np.newaxis] + delta_T - transf_transl[0]) - delta_T
                smplx_param_A_old[:3] = transl[0]

            with open(result_path / 'person2.pkl', 'rb') as f:
                data = pickle.load(f)
                motions = data['motion']
                smplx_param_B = rollout_primitives(motions)
                smplx_param_B_old = smplx_param_B[-1].copy()

                betas_B = motions[0]['betas']

                smplx_dict = {
                    'betas': betas_B[np.newaxis],
                    'transl': smplx_param_B[-1, :3][np.newaxis],
                    'global_orient': smplx_param_B[-1, 3:6][np.newaxis],
                    'body_pose': smplx_param_B[-1, 6:69][np.newaxis],
                    'return_verts': True
                }
                smplx_dict = params2torch(smplx_dict)
                smplhout = bm(**smplx_dict)
                joint_pos_B = smplhout.joints.detach().cpu().numpy()
                marker_pos_B = smplhout.vertices[:, marker_ssm_67, :].detach().cpu().numpy()

                "Comment out this because we need only one canonicalization reference. Either person1 or person2."
                # transf_rotmat, transf_transl = get_new_coordinate(smplhout)

                marker_pos_B = np.einsum('ij,bpj->bpi', transf_rotmat[0].T, marker_pos_B - transf_transl)
                joint_pos_B = np.einsum('ij,bpj->bpi', transf_rotmat[0].T, joint_pos_B - transf_transl)

                # init_B is for copying the generated motion to it.
                init_B = copy.deepcopy(data['motion'][-1])
                init_B['transf_rotmat'] = transf_rotmat
                init_B['transf_transl'] = transf_transl
                init_B['mp_type'] = '0-frame'

                "Convert the global orientation and transl of the last motion of the two persons to the canonical ones. " \
                "This conversion is for the following SLERP and alignment."
                delta_T = bm(betas=torch.FloatTensor(motions[0]['betas']).repeat(1, 1).cuda()).joints[
                          :, 0, :].detach().cpu().numpy()
                ### get new global_orient
                global_ori = Rotation.from_rotvec(
                    smplx_param_B_old[3:6][np.newaxis]).as_matrix()  # to [t,3,3] rotation mat
                global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat[0].T, global_ori)
                smplx_param_B_old[3:6] = Rotation.from_matrix(global_ori_new).as_rotvec()[0]
                ### get new transl
                transl = np.einsum('ij,tj->ti', transf_rotmat[0].T,
                                   smplx_param_B_old[:3][np.newaxis] + delta_T - transf_transl[0]) - delta_T
                smplx_param_B_old[:3] = transl[0]

            "Not using it because it will lead to persons static in the scene. Instead, by generating diffferent frames, " \
            "the two persons will move the same frame numbers."
            # "Padding and interpolate, ensure the same length and smooth transition."
            # max_frame = np.array([len(frames) for frames in [smplx_param_A, smplx_param_B]]).max()
            #
            # with open(result_path / 'person1.pkl', 'rb') as f:
            #     data = pickle.load(f)
            #     init_A_pad = copy.deepcopy(data['motion'][-1])
            #     init_A_pad['smplx_params'] = np.tile(init_A_pad['smplx_params'][0, -1:, :], (max_frame - smplx_param_A.shape[0], 1))[np.newaxis]
            #     if max_frame - smplx_param_A.shape[0] != 0:
            #         data['motion'].append(init_A_pad)
            # with open(result_path / 'person1.pkl', 'wb') as f:
            #     pickle.dump(data, f)
            #
            # with open(result_path / 'person2.pkl', 'rb') as f:
            #     data = pickle.load(f)
            #     init_B_pad = copy.deepcopy(data['motion'][-1])
            #     init_B_pad['smplx_params'] = np.tile(init_B_pad['smplx_params'][0, -1:, :], (max_frame - smplx_param_B.shape[0], 1))[np.newaxis]
            #     if max_frame - smplx_param_B.shape[0] != 0:
            #         data['motion'].append(init_B_pad)
            # with open(result_path / 'person2.pkl', 'wb') as f:
            #     pickle.dump(data, f)

            "Calculate Scene SDF"
            object_sdf = scene_sdf(scene_path, scene_sdf_path)

            sdf_points_extents = 3.  # InterGen dataset motion maximum extent is 6.3149.
            ceiling_height = 3.
            sdf_points_res = 128

            x = torch.linspace(-sdf_points_extents, sdf_points_extents, sdf_points_res)
            y = torch.linspace(-sdf_points_extents, sdf_points_extents, sdf_points_res)
            z = torch.linspace(-ceiling_height, ceiling_height, sdf_points_res)

            x, y, z = torch.meshgrid(x, y, z)
            sdf_box = torch.stack([x, y, z], dim=-1)

            # Covert to world space.
            sdf_box_w = (torch.einsum('ij,pj->pi', torch.from_numpy(transf_rotmat[0]).type(torch.float32),
                                      sdf_box.reshape(-1, 3)) + torch.from_numpy(transf_transl[0]).type(
                torch.float32)).reshape(sdf_points_res, sdf_points_res, sdf_points_res, 3)

            sdf_values = calc_sdf(sdf_box_w.reshape(-1, 3).unsqueeze(0), object_sdf).reshape(sdf_points_res,
                                                                                             sdf_points_res,
                                                                                             sdf_points_res)
            sdf_values[sdf_values >= 0] = 1
            sdf_values[sdf_values < 0] = -1

            # Ensure the points below the flow are -1. Also interpolate one interval distance.
            "Note that this is quite important, or the motion will distort quite a lot."
            sdf_values[(sdf_box_w[:, :, :, 2]) <= 0 - sdf_points_extents * 2 / 128] = -1

            # "Visualize."
            # sdf_points_vis = sdf_box_w[::8, ::8, ::8, :3]
            # sdf_points_color_judge = sdf_values[::8, ::8, ::8]
            # sdf_points_color_judge = sdf_points_color_judge.reshape(-1, 1)
            # sdf_points_vis = sdf_points_vis.reshape(-1, 3).detach().numpy()
            #
            # points_sm = []
            # for i in tqdm(range(0, len(sdf_points_vis), 1)):
            #     tfs = np.eye(4)
            #     tfs[:3, 3] = sdf_points_vis[i]
            #
            #     sm = trimesh.creation.uv_sphere(radius=0.01, transform=tfs)
            #     if sdf_points_color_judge[i] == 1:
            #         sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
            #     elif sdf_points_color_judge[i] == -1:
            #         sm.visual.vertex_colors = [0.1, 0.9, 0.1, 1.0]
            #
            #     points_sm.append(sm)
            #
            # mesh = trimesh.load(scene_path, force='mesh')
            #
            # mesh_smpl = []
            #
            # vis_model = smplx.create(r'D:\Motion\Envs\smplx\models', model_type='smplh', gender='neutral', ext='pkl',
            #                          num_betas=10, batch_size=1)
            # smplh_vis = vis_model(return_vertices=True, betas=torch.FloatTensor(betas_A[np.newaxis]),
            #                       body_pose=torch.FloatTensor(smplx_param_A[-1][6:69][None]),
            #                       global_orient=torch.FloatTensor(smplx_param_A[-1][3:6][None]),
            #                       transl=torch.FloatTensor(smplx_param_A[-1][:3][None]))
            # smplh_vis2 = vis_model(return_vertices=True, betas=torch.FloatTensor(betas_B[np.newaxis]),
            #                       body_pose=torch.FloatTensor(smplx_param_B[-1][6:69][None]),
            #                       global_orient=torch.FloatTensor(smplx_param_B[-1][3:6][None]),
            #                       transl=torch.FloatTensor(smplx_param_B[-1][:3][None]))
            #
            # for v1, v2 in zip(smplh_vis.vertices, smplh_vis2.vertices):
            #     mesh_smpl.append(trimesh.Trimesh(vertices=v1.detach().numpy(), faces=vis_model.faces, process=False))
            #     mesh_smpl.append(trimesh.Trimesh(vertices=v2.detach().numpy(), faces=vis_model.faces, process=False))
            #
            # trimesh.util.concatenate([mesh] + points_sm + mesh_smpl).show()

            sdf_condition = torch.cat([sdf_box, sdf_values.unsqueeze(-1)], dim=-1).permute(3, 0, 1, 2).unsqueeze(0)

            marker_condition = torch.cat([torch.from_numpy(marker_pos_A).reshape(1, 1, 67 * 3),
                                          torch.from_numpy(marker_pos_B).reshape(1, 1, 67 * 3)], dim=-1)

            marker_condition = torch.cat([torch.cat([torch.from_numpy(marker_pos_A) - torch.from_numpy(joint_pos_A)[:, [0]],
                                                     torch.from_numpy(joint_pos_A)[:, [0]]], dim=1).reshape(1, 1, 68 * 3),
                                torch.cat([torch.from_numpy(marker_pos_B) - torch.from_numpy(joint_pos_B)[:, [0]],
                                           torch.from_numpy(joint_pos_B)[:, [0]]], dim=1).reshape(1, 1, 68 * 3)], dim=-1)

            "Decide whether to use intergen_result for comparisons."
            if intergen_result is not None or commdm_result is not None:
                if intergen_result is not None:
                    print("Run InterGen.")
                    command = f'python D:/Motion/InterGen/tools/infer.py --prompt "{text}"'
                    print(command)
                    subprocess.run(command)
                    print("Load InterGen results.")
                    with open(intergen_result, 'rb') as f:
                        # data: [smplx_params_A: shape [103], smplx_params_B]
                        data = pickle.load(f)
                        out_params_A = data[0].numpy()
                        out_params_B = data[1].numpy()
                else:
                    print("Load commdm results.")
                    command = f'python D:/Motion/priorMDM/sample/two_person_text2motion.py ' \
                              f'--model_path D:/Motion/priorMDM/save/my_pw3d_text/model000400017.pt --text_prompt "{text}"'
                    print(command)
                    subprocess.run(command)
                    print("Load commdm results.")
                    with open(commdm_result, 'rb') as f:
                        # data: [smplx_params_A: shape [103], smplx_params_B]
                        data = pickle.load(f)
                        out_params_A = data[0].numpy()
                        out_params_B = data[1].numpy()

                # Calculate the relative position of the two people in advance.
                relative_rot = Rotation.from_rotvec(out_params_A[:, 3:6]).as_matrix().transpose(
                    (0, 2, 1)) @ Rotation.from_rotvec(out_params_B[:, 3:6]).as_matrix()
                relative_transl = Rotation.from_rotvec(out_params_A[:, 3:6]).as_matrix().transpose((0, 2, 1)) @ \
                                  (out_params_B[:, :3] - out_params_A[:, :3])[..., np.newaxis]
            else:
                "Generate HHI interactions & Retrieve corresponding hand parameters from Inter-X dataset."
                out_params, hand_params_1, hand_params_2 = pipeline_merge(sdf_condition, text, marker_condition,
                                                                          betas=None, hand_pose_retrieval=hand_pose_retrieval)
                "Directly use existing betas will lead to motion distortion in some cases."
                # betas=torch.from_numpy(np.stack([betas_A[np.newaxis], betas_B[np.newaxis]], axis=0)))

                if hand_pose_retrieval:
                    out_params_A = np.concatenate([out_params[0], hand_params_1], axis=-1)
                    out_params_B = np.concatenate([out_params[1], hand_params_2], axis=-1)
                else:
                    out_params_A = out_params[0]
                    out_params_B = out_params[1]

            "Do Alignment and SLERP."
            align_full_bodies = False if (intergen_result is None and commdm_result is None) else True
            do_slerp = True
            slerp_window_size = 4

            for n, (out_params, smplx_params_old) in enumerate(
                    zip([out_params_A, out_params_B], [smplx_param_A_old, smplx_param_B_old])):
                begin_second_motion = 0
                begin_second_motion += slerp_window_size if do_slerp else 0
                # last motion + 1 / to be used with slice

                # # Only align one of the two motions, while keep the relative position of the other.
                # if (intergen_result is not None or commdm_result is not None) and n == 0:
                #     out_params[..., 2] = smplx_params_old[2]

                # Only align the transl Z axis.
                diff_height = smplx_params_old[:3].reshape(3)[2] - out_params[..., :3].reshape(-1, 3)[0][2]
                out_params[..., :3] = out_params[..., :3] + np.array([0, 0, diff_height])

                if align_full_bodies:
                    # For the second person, leverage the relative rot and tranl to transform it. Thus can keep the original relation between the two persons.
                    if (intergen_result is not None or commdm_result is not None) and n != 0:
                        out_params[..., :3] = (Rotation.from_rotvec(
                            out_params_A[:, 3:6]).as_matrix() @ relative_transl).squeeze(-1) + out_params_A[:, :3]
                        out_params[..., 3:6] = Rotation.from_matrix(
                            Rotation.from_rotvec(out_params_A[:, 3:6]).as_matrix() @ relative_rot).as_rotvec()
                    else:
                        outputs = aligining_bodies(last_pose=torch.from_numpy(smplx_params_old[3:69]).view(22, 3),
                                                   last_trans=torch.from_numpy(smplx_params_old[:3]).view(3),
                                                   poses=torch.from_numpy(
                                                       out_params[..., 3:69][begin_second_motion:]).view(-1, 22, 3),
                                                   transl=torch.from_numpy(
                                                       out_params[..., :3][begin_second_motion:]).view(-1, 3),
                                                   pose_rep="axisangle")
                        # Alignement
                        out_params[..., 3:69][begin_second_motion:] = outputs[0].view(-1, 66).numpy()
                        out_params[..., :3][begin_second_motion:] = outputs[1].view(-1, 3).numpy()

                # Slerp if needed
                if do_slerp:
                    inter_pose = slerp_poses(last_pose=torch.from_numpy(smplx_params_old[3:69]).view(22, 3),
                                             new_pose=torch.from_numpy(out_params[..., 3:69][begin_second_motion]).view(
                                                 22, 3),
                                             number_of_frames=slerp_window_size, pose_rep="axisangle")

                    inter_transl = slerp_translation(torch.from_numpy(smplx_params_old[:3]).view(3),
                                                     torch.from_numpy(out_params[..., :3][begin_second_motion]).view(3),
                                                     number_of_frames=slerp_window_size)

                    # Fill the gap
                    out_params[..., 3:69][:begin_second_motion] = inter_pose.view(slerp_window_size, 66).numpy()
                    out_params[..., :3][:begin_second_motion] = inter_transl.view(slerp_window_size, 3).numpy()
                    if hand_pose_retrieval:
                        inter_hand_pose_start = slerp_poses(
                            last_pose=torch.cat([bm.left_hand_mean, bm.right_hand_mean], dim=-1).view(30, 3).detach().cpu(),
                            new_pose=torch.from_numpy(out_params[..., -90:][begin_second_motion]).view(30, 3),
                            number_of_frames=slerp_window_size, pose_rep="axisangle")
                        # Need to turn back to the default hand pose, as other module than HHI will not consider the hand pose.
                        inter_hand_pose_end = slerp_poses(
                            last_pose=torch.from_numpy(out_params[..., -90:][-begin_second_motion]).view(30, 3),
                            new_pose=torch.cat([bm.left_hand_mean, bm.right_hand_mean], dim=-1).view(30, 3).detach().cpu(),
                            number_of_frames=slerp_window_size, pose_rep="axisangle")
                        out_params[..., -90:][:begin_second_motion] = inter_hand_pose_start.view(slerp_window_size, 90).numpy()
                        out_params[..., -90:][-slerp_window_size:] = inter_hand_pose_end.view(slerp_window_size, 90).numpy()

            with open(result_path / 'person1.pkl', 'rb') as f:
                data = pickle.load(f)
                if hand_pose_retrieval:
                    init_A['smplx_params'] = np.concatenate([out_params_A[np.newaxis][..., :-10-90], out_params_A[np.newaxis][..., -90:]], axis=-1)
                else:
                    init_A['smplx_params'] = out_params_A[np.newaxis][..., :-10]
                data['motion'].append(init_A)
                "Update the betas."
                pelvis = bm(betas=torch.FloatTensor(data['motion'][0]['betas']).unsqueeze(0).cuda()).joints[:, 0, :].detach().cpu().numpy()
                if use_predicted_betas:
                    for mot in data['motion']:
                        mot['betas'] = out_params_A[0, -10-90:-90] if hand_pose_retrieval else out_params_A[0, -10:]
                data['motion'][-1]['pelvis_loc'] = data['motion'][-1]['smplx_params'][[0], :, :3] + pelvis
            with open(result_path / 'person1.pkl', 'wb') as f:
                pickle.dump(data, f)

            with open(result_path / 'person2.pkl', 'rb') as f:
                data = pickle.load(f)
                if hand_pose_retrieval:
                    init_B['smplx_params'] = np.concatenate([out_params_B[np.newaxis][..., :-10-90], out_params_B[np.newaxis][..., -90:]], axis=-1)
                else:
                    init_B['smplx_params'] = out_params_B[np.newaxis][..., :-10]
                data['motion'].append(init_B)
                pelvis = bm(betas=torch.FloatTensor(data['motion'][0]['betas']).unsqueeze(0).cuda()).joints[:, 0, :].detach().cpu().numpy()
                if use_predicted_betas:
                    for mot in data['motion']:
                        mot['betas'] = out_params_B[0, -10-90:-90] if hand_pose_retrieval else out_params_A[0, -10:]
                data['motion'][-1]['pelvis_loc'] = data['motion'][-1]['smplx_params'][[0], :, :3] + pelvis
            with open(result_path / 'person2.pkl', 'wb') as f:
                pickle.dump(data, f)

            # Reset the flag.
            human_inter_flag = False
            "----------------------------------------------------------------------------------------"

@torch.no_grad()
def human_collision_eval(smplx_params_A, smplx_params_B, betas_A, betas_B):
    smplx_model = smplx.create(get_SMPL_SMPLH_SMPLX_body_model_path(), model_type='smplx',
                                        gender='neutral', ext='pkl',
                                        num_betas=10,
                                        num_pca_comps=12,
                                        batch_size=len(smplx_params_A)
                                        ).cuda()
    smplx_model.eval()

    bparam = {}
    bparam['transl'] = smplx_params_A[:, :3]
    bparam['global_orient'] = smplx_params_A[:, 3:6]
    bparam['body_pose'] = smplx_params_A[:, 6:69]
    bparam['betas'] = np.tile(betas_A.reshape(1, -1), (len(smplx_params_A), 1))
    vertices_A = smplx_model(return_verts=True, **params2torch(bparam)).vertices

    bparam = {}
    bparam['transl'] = smplx_params_B[:, :3]
    bparam['global_orient'] = smplx_params_B[:, 3:6]
    bparam['body_pose'] = smplx_params_B[:, 6:69]
    bparam['betas'] = np.tile(betas_B.reshape(1, -1), (len(smplx_params_B), 1))
    vertices_B = smplx_model(return_verts=True, **params2torch(bparam)).vertices

    seg = 500
    humanpeneloss_manager = GeneralContactLoss(body_model_utils_folder=get_human_penetration_essentials_path())
    humanpeneloss_manager.weight = 1
    num, loss, all_loss = 0, 0, []
    for i in range(0, len(smplx_params_A), seg):
        vertices_A_tmp = vertices_A[i:i + seg]
        vertices_B_tmp = vertices_B[i:i + seg]
        length = len(vertices_A_tmp)

        humanpeneloss_manager.forward(torch.ones(1, length, 2, 1).cuda(), torch.ones(1).cuda(), v1=vertices_A_tmp.view(1, length, -1, 3),
                                      v2=vertices_B_tmp.view(1, length, -1, 3), factor=1, return_elemtent=True)

        num_, all_loss_ = humanpeneloss_manager.losses['humanpenetration']

        num, _ = num + num_.detach().cpu().numpy(), all_loss.append(all_loss_.detach().cpu().numpy())
        loss += all_loss_.sum()

    loss = loss / (num + 1.e-7)
    all_loss = np.concatenate(all_loss, axis=0)

    index = np.where(all_loss.sum(-1) > 0.1)[0]

    index_starts, index_ends = [], []
    # Extract all the sequences that have collision, and get these sequences' start and end index.

    if len(index) != 0:
        for k, g in groupby(enumerate(index), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            index_starts.append(group[0])
            index_ends.append(group[-1])

    return index_starts, index_ends, loss

def collision_revision(smplx_params_A, smplx_params_B, betas_A, betas_B):
    # To avoid the characters' collision, suppose there is a sequence start with index_start and end with index_end,
    # then try to speed up the motions in [0, N_A] by downsampling frames and slow down motions in [N_A, -1] by upsamping frames for character A, and
    # slow down the motions in [0, N_B] and speed up motions in [N_B, -1] for character B. Then, see if the
    # collision is resolved (i.e., loss decreases). (N_A and N_B's values are set according to different situations)
    index_starts, index_ends, last_loss = human_collision_eval(smplx_params_A, smplx_params_B, betas_A, betas_B)

    optimal_smplx_params_A = smplx_params_A
    optimal_smplx_params_B = smplx_params_B

    ind = 0
    while ind < len(index_starts):
        index_start, index_end = index_starts[ind], index_ends[ind]
        ind += 1
        smplx_params_A = optimal_smplx_params_A
        smplx_params_B = optimal_smplx_params_B

        if index_start == 0 or index_end == 0 or (index_end - index_start) < 5:
            continue
        else:
            N_A = N_B = (index_start + index_end) // 2
            marginal_frames = (index_end - index_start) // 2
            # Speed up the motions in [0, N_A] by downsampling frames and slow down motions in [N_A, -1] by upsamping frames for character A
            # Slow down the motions in [0, N_B] and speed up motions in [N_B, -1] for character B.

            new_num_frames = int(N_A - marginal_frames)
            if new_num_frames < 1:
                continue
            downsample_ids_A = np.linspace(0, N_A - 1, num=new_num_frames, dtype=int)
            new_num_frames = int(len(smplx_params_A) - N_A + marginal_frames)
            if new_num_frames < 1:
                continue
            upsample_ids_A = np.linspace(N_A, len(smplx_params_A) - 1, num=new_num_frames, dtype=int)

            new_num_frames = int(N_B + marginal_frames)
            if new_num_frames < 1:
                continue
            upsample_ids_B = np.linspace(0, N_B - 1, num=new_num_frames, dtype=int)
            new_num_frames = int(len(smplx_params_B) - N_B - marginal_frames)
            if new_num_frames < 1:
                continue
            downsample_ids_B = np.linspace(N_B, len(smplx_params_B) - 1, num=new_num_frames, dtype=int)

            smplx_params_A = np.concatenate([smplx_params_A[downsample_ids_A], smplx_params_A[upsample_ids_A]])
            smplx_params_B = np.concatenate([smplx_params_B[upsample_ids_B], smplx_params_B[downsample_ids_B]])

            # Check if the collision is resolved
            index_starts_tmp, index_ends_tmp, loss = human_collision_eval(smplx_params_A, smplx_params_B, betas_A, betas_B)
            if loss < last_loss:
                optimal_smplx_params_A = smplx_params_A
                optimal_smplx_params_B = smplx_params_B
                last_loss = loss
                index_starts = index_starts_tmp
                index_ends = index_ends_tmp
                # Repeat the process
                ind = 0
            else:
                continue

    return optimal_smplx_params_A, optimal_smplx_params_B

def end_blender_file_preparation(result_path):
    """Convert the SMPLX-format motion to the SMPLH-format motion and then to BVH format, " \
        "this is for the afterward Blender visualization."""
    body_regressor = MoshRegressor().cuda()
    body_regressor.load_state_dict(
        torch.load(get_smplh_body_regressor_checkpoint_path(), map_location='cuda')['model_state_dict'])
    body_regressor.eval()
    blender_res_path = (result_path / 'smplx-bvh') if hand_pose_retrieval else (result_path / 'smplh-bvh')
    blender_res_path.mkdir(parents=True, exist_ok=True)

    with open(get_SSM_SMPLX_body_marker_path()) as f:
        markerdict = json.load(f)['markersets'][0]['indices']
    markers = list(markerdict.values())

    "=========collision revision======="
    if collision_revision_flag:
        with open(result_path / f'person1.pkl', 'rb') as f:
            data = pickle.load(f)
            motions_A = data['motion']
        with open(result_path / f'person2.pkl', 'rb') as f:
            data = pickle.load(f)
            motions_B = data['motion']
        # Break motions_A and _B into segments by the motion list that have more than 100 frames (i.e., HHI interactions).
        motion_A_segs = []
        motion_B_segs = []
        seg = []
        for mots in motions_A:
            seg.append(mots)
            if len(mots['smplx_params'][0]) > 100:
                motion_A_segs.append(seg)
                seg = []
        motion_A_segs.append(seg)
        seg = []
        for mots in motions_B:
            seg.append(mots)
            if len(mots['smplx_params'][0]) > 100:
                motion_B_segs.append(seg)
                seg = []
        motion_B_segs.append(seg)
        smplx_params_revise = [[], []]
        # Ignore the last seg, and also the last HHI interaction in the seg.
        for seg_A, seg_B in zip(motion_A_segs[:-1], motion_B_segs[:-1]):
            smplx_params_A = rollout_primitives(seg_A[:-1], hand_pose_retrieval=hand_pose_retrieval)
            smplx_params_B = rollout_primitives(seg_B[:-1], hand_pose_retrieval=hand_pose_retrieval)

            # # Clip length.
            # if len(smplx_params_A) > len(smplx_params_B):
            #     smplx_params_A = smplx_params_A[:len(smplx_params_B)]
            # else:
            #     smplx_params_B = smplx_params_B[:len(smplx_params_A)]

            # padding length.
            if len(smplx_params_A) > len(smplx_params_B):
                smplx_params_B = np.concatenate([smplx_params_B, np.tile(smplx_params_B[-1][np.newaxis], (
                len(smplx_params_A) - len(smplx_params_B), 1))], axis=0)
            else:
                smplx_params_A = np.concatenate([smplx_params_A, np.tile(smplx_params_A[-1][np.newaxis], (
                len(smplx_params_B) - len(smplx_params_A), 1))], axis=0)

            betas_A = seg_A[0]['betas']
            betas_B = seg_B[0]['betas']

            smplx_params_A_seg, smplx_params_B_seg = collision_revision(smplx_params_A, smplx_params_B, betas_A, betas_B)
            smplx_params_revise[0].append(smplx_params_A_seg)
            smplx_params_revise[0].append(rollout_primitives([seg_A[-1]], hand_pose_retrieval=hand_pose_retrieval))
            smplx_params_revise[1].append(smplx_params_B_seg)
            smplx_params_revise[1].append(rollout_primitives([seg_B[-1]], hand_pose_retrieval=hand_pose_retrieval))
        if len(motion_A_segs[-1]) > 0:
            smplx_params_revise[0].append(rollout_primitives(motion_A_segs[-1], hand_pose_retrieval=hand_pose_retrieval))
        if len(motion_B_segs[-1]) > 0:
            smplx_params_revise[1].append(rollout_primitives(motion_B_segs[-1], hand_pose_retrieval=hand_pose_retrieval))
        smplx_params_revise[0] = np.concatenate(smplx_params_revise[0], axis=0)
        smplx_params_revise[1] = np.concatenate(smplx_params_revise[1], axis=0)

    "=========final file preparation (include hand aug)======="
    for i in [1, 2]:
        with open(result_path / f'person{i}.pkl', 'rb') as f:
            data = pickle.load(f)
            motions = data['motion']
            smplx_param = rollout_primitives(motions, hand_pose_retrieval=hand_pose_retrieval)

            # Replace the intermediate motions with the revised ones.
            if collision_revision_flag:
                smplx_param = smplx_params_revise[i - 1]
            betas = motions[0]['betas']

            if hand_pose_retrieval:
                params = np.concatenate([smplx_param, betas[np.newaxis].repeat(len(smplx_param), axis=0)], axis=-1)
            else:
                smplx_dict = {
                    'betas': betas[np.newaxis].repeat(len(smplx_param), axis=0),
                    'transl': smplx_param[:, :3],
                    'global_orient': smplx_param[:, 3:6],
                    'body_pose': smplx_param[:, 6:69],
                    'return_verts': True
                }
                smplx_dict = params2torch(smplx_dict)

                bm = smplx.create(bm_path, model_type='smplx',
                                  gender='neutral', ext='pkl',
                                  create_global_orient=True,
                                  create_body_pose=True,
                                  create_betas=True,
                                  create_transl=True,
                                  batch_size=len(smplx_param)
                                  ).eval().cuda()
                smplhout = bm(**smplx_dict)
                marker_pos = smplhout.vertices[:, markers, :]

                params = body_regressor(marker_pos.reshape(-1, 67 * 3), 1, marker_pos.shape[0]).detach().cpu().numpy()

            with open(blender_res_path / f'person{i}.pkl', 'wb') as f:
                # If hand_pose_retrieval=True, the following params will have been augmented with hand poses.
                pickle.dump(params, f)

    if hand_pose_retrieval:
        command = f"python visualization/smplx2bvh.py --poses {blender_res_path} --output {blender_res_path}"
    else:
        command = f"python visualization/smpl2bvh.py --poses {blender_res_path} --output {blender_res_path}"
    subprocess.run(command)


def run(scene_name, start_iter):
    scene_dir = Path(fr'D:\Motion\Story-HIM\HSInter\data\replica\{scene_name}')
    scene_path = scene_dir / 'mesh_floor.ply'
    json_path = scene_dir / 'habitat' / 'info_semantic.json'
    scene_sdf_path = scene_dir / 'sdf' / 'scene_sdf.pkl'

    os.makedirs(scene_dir / 'sdf', exist_ok=True)

    sub_path = f"results_Story_HIM_{scene_name}_{start_iter}"

    result_path = Path(f'{weight_name}/{sub_path}')
    if result_path.exists():
        shutil.rmtree(result_path)
    result_path.mkdir(exist_ok=True)

    "======== Manually define the motion orders ==========="
    # # Define here the walking route by name of objects or None for random sampling.
    # # Example: [None, "sofa", "sit", "stand", "table", "HHI <human interaction prompt>" ...]
    # text = "In an intense boxing match, one is continuously punching while the other is defending and counterattacking."
    # input_orders_1 = [None, 'door', None, 'table', 'HHI: the two face each other and exchange a few words about the book',
    #                   'cabinet', None, 'chair', 'sit on', 'HHI: the two face each other and exchange a few words about the book.', None, 'door']
    # input_orders_2 = [None, 'door', None, 'table', 'HHI: the two face each other and exchange a few words about the book',
    #                   'sofa', 'sit on', None, None, 'HHI: the two face each other and exchange a few words about the book', 'sofa', 'lie on']
    "======================================================"

    "======== Scene preprocessing ==========="
    print("\n Start processing scene...")
    accessible_object = first_scene_preparation(json_path, scene_dir, scene_name)
    print("Finish processing scene. \n")
    "========================================="

    "======== Automatically LLM plot & order generation ==========="
    print("\n Start generating LLM plot & orders...")
    while True:
        plot = llm_order_generation(", ".join(set(accessible_object.keys())))
        if plot == "":
            # sleep
            time.sleep(5)
            continue
        else:
            input_orders_1 = plot.split("[")[1].split("]")[0]
            input_orders_2 = plot.split("[")[2].split("]")[0]

            # recheck the reasonability of the orders.
            revised_orders = llm_order_revision(input_orders_1, input_orders_2)
            if revised_orders is not None:
                # Save the plot to a txt file in result_path.
                with open(result_path / 'plot.txt', 'w') as f:
                    f.write(plot)
                # Use '|' because ',' may also split the HHI description.
                input_orders_1 = revised_orders.split("[")[1].split("]")[0].split(" | ")
                input_orders_2 = revised_orders.split("[")[2].split("]")[0].split(" | ")

                for input_orders in [input_orders_1, input_orders_2]:
                    for i in range(len(input_orders)):
                        if input_orders[i] == "None":
                            input_orders[i] = None
                break
    print(f"Finish generating LLM plot & orders. \n")
    "=============================================="

    "======== Orders processing ==========="
    print("\n Start processing orders...")
    processed_orders = orders_revision(input_orders_1, input_orders_2, accessible_object)
    print(f"Finish processing orders. Orders-1: {processed_orders[0]} \t Orders-2: {processed_orders[1]} \n")

    # Save the orders to a txt file in result_path.
    with open(result_path / 'orders.txt', 'w') as f:
        f.write(f"Orders-1: {processed_orders[0]} \n Orders-2: {processed_orders[1]}")
    "======================================="

    # if os.path.exists(os.path.join("results_intergen", sub_path)):
    #     with open(os.path.join("results_intergen", sub_path, 'orders.txt'), 'r') as f:
    #         orders = f.read()
    #         orders = orders.split("\n")
    #         input_orders_1 = ast.literal_eval(orders[0].split("Orders-1: ")[1])
    #         input_orders_2 = ast.literal_eval(orders[1].split("Orders-2: ")[1])
    #
    #         # # remove orders with words "HHI:"
    #         # input_orders_1 = [order for order in input_orders_1 if order is None or "HHI:" not in order]
    #         # input_orders_2 = [order for order in input_orders_2 if order is None or "HHI:" not in order]
    #
    #         processed_orders = [input_orders_1, input_orders_2]

    "======== Navmesh preprocessing ==========="
    print("\n Start processing navmesh...")
    floor_height = 0
    navmesh_tight_path = scene_dir / 'navmesh_tight.ply'
    navmesh_loose_path = scene_dir / 'navmesh_loose.ply'
    # get tight navmesh for local map sensing
    navmesh_tight = get_navmesh(navmesh_tight_path, scene_path, agent_radius=0.05, floor_height=floor_height,
                                visualize=visualize)
    # get loose navmesh for path planning
    navmesh_loose = get_navmesh(navmesh_loose_path, scene_path, agent_radius=0.15, floor_height=floor_height,
                                visualize=visualize)
    print("Finish processing navmesh. \n")
    "============================================"

    "======== Orders segmentation ========"
    print("\n Start segmenting orders...")
    segmented_orders = orders_segmentation(processed_orders)
    print(f"Finish segmenting orders. Segments-1: {segmented_orders[0]} \t Segments-2: {segmented_orders[1]} \n")
    "======================================="

    "======== Automatic path sampling and planning ==========="
    print("\n Start automatic path sampling and planning...")
    path_name = f'{scene_name}_path'
    wpath_path = scene_dir / f'{path_name}.pkl'
    scene_mesh = trimesh.load(scene_path, force='mesh')
    extents = scene_mesh.bounding_box.extents
    centroid = scene_mesh.bounding_box.centroid
    all_wpaths, append_vars = walking_path_points_generation(scene_dir, scene_name, scene_path, segmented_orders,
                                                             accessible_object, navmesh_loose,
                                                             floor_height=floor_height,
                                                             visualize=visualize, extents=extents, centroid=centroid,
                                                             scene_mesh=scene_mesh)
    print(f"Finish automatic path sampling and planning. \n")
    "======================================="

    "======= Split inputs according to the HHI text ======="
    print("\n Start splitting inputs according to the HHI text...")
    humaninter_segs = mid_further_split_orders(all_wpaths, segmented_orders, append_vars)
    print(f"Finish splitting inputs according to the HHI text. \n")
    "======================================="

    "======== Automatic human locomotion, human-object interaction and human-human interaction ==========="
    print("\n Start automatic human locomotion, human-object interaction and human-human interaction...")
    motion_generation(humaninter_segs, result_path, scene_path, scene_name, navmesh_tight_path, navmesh_loose, scene_mesh,
                      wpath_path, path_name, scene_sdf_path, floor_height=floor_height)
    print("Finish automatic human locomotion, human-object interaction and human-human interaction. \n")
    "======================================="

    "Additionally convert the SMPLX-format motion to the SMPLH-format motion and then to BVH format, " \
    "this is for the following Blender visualization."
    print("\n Start converting the SMPLX-format motion to the SMPLH-format motion and then to BVH format...")
    end_blender_file_preparation(result_path)
    print("Finish converting the SMPLX-format motion to the SMPLH-format motion and then to BVH format. \n")

    "Visualize the results."
    command = f"python vis_gen.py --seq_path {result_path / 'person*.pkl'}"
    subprocess.run(command)


if __name__ == "__main__":
    visualize = False

    weight_name = "./results-story"
    # make dirs
    os.makedirs(weight_name, exist_ok=True)

    # intergen_result = r"D:\Motion\InterGen\results\In an intense boxing match, one is continuously punching while the other is defending and counterattacking..pkl"
    intergen_result = None # r"D:\Motion\InterGen\results\for_story.pkl"  # Set to the path of the InterGen result file if want to compare with InterGen. The result format should be indentical to smpl_params from the regressor.
    commdm_result = None # r"D:\Motion\priorMDM\for_story.pkl"
    use_predicted_betas = False  # If True, use the betas predicted by body regressor.
    hand_pose_retrieval = True  # If True, use the hand pose retrieval to get the hand pose.
    collision_revision_flag = True  # If True, revise the collision by speed up and slow down the motions.
    assert not ((intergen_result or commdm_result) and hand_pose_retrieval), "Hand_pose_retrieval only support Story-HIM."
    assert not (intergen_result and commdm_result), "Only one of the intergen_result and commdm_result can be set."

    scene_names = [name for name in os.listdir(r'D:\Motion\Story-HIM\HSInter\data\replica') if "sdf" not in name]
    for id, scene_name in tqdm(enumerate(scene_names)):
        start_iter = 0
        while start_iter < 10:
            # try:
            run(scene_name, start_iter)
            # except Exception as e:
            #     # print error
            #     print("Error:", e)
            #     continue
            start_iter += 1
