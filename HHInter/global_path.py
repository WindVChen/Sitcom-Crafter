def get_SMPL_SMPLH_SMPLX_body_model_path():
    bmpath = r'D:\Motion\Envs\smplx\models'
    return bmpath

def get_SSM_SMPL_body_marker_path():
    mkpath = r'D:\Motion\Story-HIM\HSInter\data\models_smplx_v1_1\models\markers\SSM-smplh.json'
    return mkpath

def get_SSM_SMPLX_body_marker_path():
    mkpath = r'D:\Motion\Story-HIM\HSInter\data\models_smplx_v1_1\models\markers\SSM2.json'
    return mkpath

def get_smplx_body_regressor_checkpoint_path():
    ckpath = r"D:\Motion\Story-HIM\marker_regressor\results-smplx-reg\exp_GAMMAPrimitive\MoshRegressor_v3_neutral_new\checkpoints\epoch-150.ckp"
    return ckpath

def get_smplh_body_regressor_checkpoint_path():
    ckpath = r"D:\Motion\Story-HIM\marker_regressor\results-smplh\exp_GAMMAPrimitive\MoshRegressor_v3_neutral_new\checkpoints\epoch-300.ckp"
    return ckpath

def get_human_penetration_essentials_path():
    path = r'D:\Motion\Story-HIM\HHInter\data\essentials\body_model_utils'
    return path