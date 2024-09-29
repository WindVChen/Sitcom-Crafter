import torch
import numpy as np
import random
import glob
import os, sys
import smplx
from scipy.spatial.transform import Rotation
from torch.utils import data


sys.path.append(os.getcwd())


class BatchGeneratorAMASSCanonicalized(data.Dataset):
    def __init__(self,
                data_path, seq_len, body_model_path, data_list
                ):
        self.rec_list = list()
        self.data_path = data_path
        self.seq_len = seq_len

        self.body_model = smplx.create(body_model_path, model_type='smplh',
                                       gender='neutral', ext='pkl',
                                       num_betas=10,
                                       batch_size=1)

        if isinstance(data_list, list):
            self.data_list = []
            for data in data_list:
                with open(data, "r") as f:
                    self.data_list += f.readlines()
        elif isinstance(data_list, str):
            self.data_list = open(data_list, "r").readlines()
        else:
            raise ValueError("data_list must be a list or a path to a list of files")

        for file in glob.glob(os.path.join(self.data_path, 'person*/*.npz')):
            if os.path.basename(file).split(".")[0].split("_")[0] + "\n" in self.data_list:
                with np.load(file) as data:
                    if len(data['marker_ssm2_67']) >= self.seq_len:
                        self.rec_list.append(file)

        print(f"Dataset includes {len(self.rec_list)} recordings!")

    def __len__(self):
        return len(self.rec_list)

    def __getitem__(self, item):
        rec = self.rec_list[item]

        with np.load(rec) as data:
            body_ssm2_67 = data['marker_ssm2_67']
            pose = data['poses']
            betas = data['betas']
            transl = data['trans']
            canonical_ratio = 0.8
            is_canonicalize = random.random() < canonical_ratio
            if is_canonicalize:
                canonical_center_idx = random.choice(range(0, len(body_ssm2_67), 1))
                transf_rotmat, transf_transl = data['transf_rotmat'][canonical_center_idx], data['transf_transl'][canonical_center_idx]
                delta_T = self.body_model(betas=torch.FloatTensor(betas).repeat(1, 1)).joints[
                          :, 0, :].detach().cpu().numpy()
                ### get new global_orient
                global_ori = Rotation.from_rotvec(pose[:, :3]).as_matrix()  # to [t,3,3] rotation mat
                global_ori_new = np.einsum('ij,tjk->tik', transf_rotmat.T, global_ori)
                pose[:, :3] = Rotation.from_matrix(global_ori_new).as_rotvec()
                ### get new transl
                transl = np.einsum('ij,tj->ti', transf_rotmat.T, transl + delta_T - transf_transl) - delta_T
                body_ssm2_67 = np.einsum('ij,bpj->bpi', transf_rotmat.T, body_ssm2_67 - transf_transl)

        ## normalized walking path and unnormalized marker to target
        body_feature = body_ssm2_67.reshape([-1,67*3])

        if len(body_feature) > self.seq_len:
            idx = random.choice(list(range(0, len(body_feature) - self.seq_len, 1)))
            body_feature = body_feature[idx:idx+self.seq_len]
            pose = pose[idx:idx+self.seq_len]
            transl = transl[idx:idx+self.seq_len]

        return body_feature.astype(np.float32), betas.astype(np.float32), transl.astype(np.float32), pose[:, :3].astype(np.float32), pose[:, 3:].astype(np.float32)

