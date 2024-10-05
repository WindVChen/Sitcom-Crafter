import os.path
import bpy
import pickle
from HHInter.global_path import get_program_root_path


def visualize(mesh_path, result_path):
    with open(os.path.join(result_path, "person1.bvh"), 'rb') as f:
        bvh_data = f.readlines()
    for i in bvh_data:
        if "Frames" in i.decode():
            frame_num = int(i.decode().split(" ")[1])
            break

    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.mode_set(mode='OBJECT')
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.delete(use_global=False, confirm=False)
    scene = bpy.data.scenes['Scene']
    scene.render.fps = 30
    scene.frame_end = frame_num

    "person 1"
    bpy.ops.import_anim.bvh(filepath=os.path.join(result_path, "person1.bvh"), axis_forward='Y', axis_up='Z')
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.import_scene.fbx(filepath=os.path.join(other_elements_path, "Ch37_nonPBR.fbx" if character_pair == 1 else "Ch22_nonPBR.fbx"))
    bpy.ops.object.mode_set(mode='POSE')

    charact = "6" if character_pair == 1 else "2"
    bpy.context.scene.keemap_settings.bone_mapping_file = os.path.join(other_elements_path,
                                                                       f"mapping{charact}new.json" if file_type == 'smplh' else f"smplx-mapping{charact}new-final.json")
    bpy.ops.wm.keemap_read_file()

    bpy.context.scene.keemap_settings.source_rig_name = "person1"
    bpy.context.scene.keemap_settings.destination_rig_name = "Armature"
    bpy.context.scene.keemap_settings.number_of_frames_to_apply = frame_num

    bpy.ops.wm.perform_animation_transfer()

    "person 2"
    bpy.ops.import_anim.bvh(filepath=os.path.join(result_path, "person2.bvh"), axis_forward='Y', axis_up='Z')
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.import_scene.fbx(filepath=os.path.join(other_elements_path, "Ch31_nonPBR.fbx" if character_pair == 1 else "Ch23_nonPBR.fbx"))  # Ch23
    bpy.ops.object.mode_set(mode='POSE')

    charact = "9" if character_pair == 1 else ""
    bpy.context.scene.keemap_settings.bone_mapping_file = os.path.join(other_elements_path,
                                                                       f"mapping{charact}new.json" if file_type == 'smplh' else f"smplx-mapping{charact}new-final.json")
    bpy.ops.wm.keemap_read_file()

    bpy.context.scene.keemap_settings.source_rig_name = "person2"
    bpy.context.scene.keemap_settings.destination_rig_name = "Armature.001"
    bpy.context.scene.keemap_settings.number_of_frames_to_apply = frame_num

    bpy.ops.wm.perform_animation_transfer()

    "scene mesh"
    bpy.ops.wm.ply_import(filepath=mesh_path)

    my_areas = bpy.context.workspace.screens[0].areas
    my_shading = 'SOLID'  # 'WIREFRAME' 'SOLID' 'MATERIAL' 'RENDERED'

    for area in my_areas:
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.shading.type = my_shading
                space.shading.show_backface_culling = True

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.mode_set(mode='OBJECT')


if __name__ == "__main__":
    character_pair = 1 # 2
    file_type = "smplx"  # smplh / smplx, where smplx has hand pose.
    other_elements_path = os.path.join(get_program_root_path(), "Sitcom-Crafter/HSInter/visualization/data")

    sub_folders = ["results_Story_HIM_apartment_1_1", "results_Story_HIM_office_0_0", "results_Story_HIM_office_0_1", "results_Story_HIM_office_0_2", "results_Story_HIM_office_1_0", "results_Story_HIM_office_1_1", "results_Story_HIM_office_3_2", "results_Story_HIM_office_4_1", "results_Story_HIM_room_0_1", "results_Story_HIM_room_1_2", "results_Story_HIM_room_2_0", "results_Story_HIM_room_2_1", "results_Story_HIM_room_2_2"]

    for id in [1, 2]:
        character_pair = id
        for sub_folder in sub_folders:
            root_path = os.path.join(os.path.join(get_program_root_path(), "Sitcom-Crafter/HSInter/Results-custom-story"), sub_folder)
            result_path = os.path.join(root_path, f"{file_type}-bvh")
            with open(os.path.join(root_path, "person1.pkl"), 'rb') as f:
                mesh_path = pickle.load(f)['scene_path']
            # PATH format to str format
            mesh_path = str(mesh_path)
            visualize(mesh_path, result_path)

            # save blender main file
            bpy.ops.wm.save_as_mainfile(filepath=sub_folder + '_' + str(id) + '_anime.blend')

            # remove/purge all the objects
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()