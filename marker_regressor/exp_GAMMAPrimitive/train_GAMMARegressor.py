import os
import sys
import argparse
import torch
import numpy as np
sys.path.append(os.getcwd())
from marker_regressor.exp_GAMMAPrimitive.utils.config_creator import ConfigCreator
from marker_regressor.exp_GAMMAPrimitive.utils.batch_gen_amass import BatchGeneratorAMASSCanonicalized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="MoshRegressor_v3_neutral_new")
    parser.add_argument('--resume_training', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    args = parser.parse_args()

    """load the right model"""
    from marker_regressor.models.models_GAMMA_primitive import GAMMARegressorTrainOP

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    cfgall = ConfigCreator(args.cfg)
    modelcfg = cfgall.modelconfig
    losscfg = cfgall.lossconfig
    traincfg = cfgall.trainconfig

    traincfg['resume_training'] = True if args.resume_training==1 else False
    traincfg['verbose'] = True if args.verbose==1 else False
    traincfg['gpu_index'] = args.gpu_index

    data_list = [os.path.join(traincfg['dataset_path'], 'train.txt'), os.path.join(traincfg['dataset_path'], 'val.txt')]

    """data"""
    train_dataset = BatchGeneratorAMASSCanonicalized(data_path=traincfg['dataset_path'], seq_len=modelcfg['seq_len'],
                                                 body_model_path=traincfg['body_model_path'], data_list=data_list)
    trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=traincfg['batch_size'],
            num_workers=traincfg['num_workers'],
            pin_memory=False,
            shuffle=True,
            drop_last=True,
            persistent_workers=True
            )

    """model and trainop"""
    trainop = GAMMARegressorTrainOP(modelcfg, losscfg, traincfg)
    trainop.train(trainloader)

