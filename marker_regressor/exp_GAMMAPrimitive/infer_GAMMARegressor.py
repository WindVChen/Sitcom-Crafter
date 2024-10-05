import os
import sys
import argparse
import torch
import numpy as np
sys.path.append(os.getcwd())
from marker_regressor.exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from marker_regressor.exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized
import random
from HHInter.global_path import get_program_root_path


def seed_torch(seed=0):
    print("Seed Fixed!")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="MoshRegressor_v3_neutral_new")
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default=os.path.join(get_program_root_path(), "Sitcom-Crafter/marker_regressor/results-smplh/exp_GAMMAPrimitive/MoshRegressor_v3_neutral_new/checkpoints/epoch-300.ckp"))
    args = parser.parse_args()

    """load the right model"""
    from marker_regressor.models.models_GAMMA_primitive import GAMMARegressorInferOP

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfgall = ConfigCreator(args.cfg)
    modelcfg = cfgall.modelconfig
    losscfg = cfgall.lossconfig
    traincfg = cfgall.trainconfig

    traincfg['verbose'] = True if args.verbose==1 else False
    traincfg['gpu_index'] = args.gpu_index
    traincfg['checkpoint_path'] = args.checkpoint_path
    traincfg['batch_size'] = 1

    modelcfg['seq_len'] = 1

    data_list = os.path.join(traincfg['dataset_path'], 'test.txt')

    """data"""
    test_dataset = BatchGeneratorAMASSCanonicalized(data_path=traincfg['dataset_path'], seq_len=modelcfg['seq_len'],
                                                     body_model_path=traincfg['body_model_path'], data_list=data_list)
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False,
        shuffle=False,
        drop_last=False,
        persistent_workers=False
    )

    """model and trainop"""
    seed_torch(0)
    inferop = GAMMARegressorInferOP(modelcfg, losscfg, traincfg)
    inferop.infer(testloader)

