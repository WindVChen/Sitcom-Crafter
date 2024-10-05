# This is the root path of the dataset. That is InterGen and Inter-X datasets are located in this path. Also with some smplx and smplh body models under this path.
def get_dataset_path():
    path = r"E:\Dataset\HumanMotion"
    return path

# This is the root path of the program. That is the Sitcom-Crafter project is located in this path.
def get_program_root_path():
    path = r"E:\ComputerPrograms\HumanMotion"
    return path

# Path where you save the SMPL/SMPLH/SMPLX body model weight files.
def get_SMPL_SMPLH_SMPLX_body_model_path():
    bmpath = r'E:\Dataset\HumanMotion\smplx\models'
    return bmpath

# Path where you save the SSM-smplh.json file that describes information of marker points.
def get_SSM_SMPL_body_marker_path():
    mkpath = r'E:\ComputerPrograms\HumanMotion\Sitcom-Crafter\HSInter\data\models_smplx_v1_1\models\markers\SSM-smplh.json'
    return mkpath

# Path where you save the SSM-smplx.json file that describes information of marker points.
def get_SSM_SMPLX_body_marker_path():
    mkpath = r'E:\ComputerPrograms\HumanMotion\Sitcom-Crafter\HSInter\data\models_smplx_v1_1\models\markers\SSM2.json'
    return mkpath

# Path where you save the pretrained model of the marker regressor for smplx body model.
def get_smplx_body_regressor_checkpoint_path():
    ckpath = r"E:\ComputerPrograms\HumanMotion\Sitcom-Crafter\marker_regressor\results-smplx-reg\exp_GAMMAPrimitive\MoshRegressor_v3_neutral_new\checkpoints\epoch-150.ckp"
    return ckpath

# Path where you save the pretrained model of the marker regressor for smplh body model.
def get_smplh_body_regressor_checkpoint_path():
    ckpath = r"E:\ComputerPrograms\HumanMotion\Sitcom-Crafter\marker_regressor\results-smplh\exp_GAMMAPrimitive\MoshRegressor_v3_neutral_new\checkpoints\epoch-300.ckp"
    return ckpath

# Path where you save the essential files for human penetration.
def get_human_penetration_essentials_path():
    path = r'E:\ComputerPrograms\HumanMotion\Sitcom-Crafter\HHInter\data\essentials\body_model_utils'
    return path