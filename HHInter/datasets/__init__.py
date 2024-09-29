import lightning.pytorch as pl
import torch
from .interhuman import InterHumanDataset
from .evaluator import (
    EvaluatorModelWrapper,
    EvaluationDataset,
    get_dataset_motion_loader,
    get_motion_loader)
from functools import partial
# from .dataloader import build_dataloader

__all__ = [
    'InterHumanDataset', 'EvaluationDataset',
    'get_dataset_motion_loader', 'get_motion_loader']

def build_loader(cfg, data_cfg):
    # setup data
    if data_cfg.NAME == "interhuman":
        train_dataset = InterHumanDataset(data_cfg)
    else:
        raise NotImplementedError

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=1,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        )

    return loader

def collate_fn(batch, device):
    "return name, text, motion1, motion2, gt_length - self.cond_length, self.cond_length, \
            np.concatenate((motion1_transf_rotmat, motion2_transf_rotmat), axis=-1), \
            np.concatenate((motion1_transf_transl, motion2_transf_transl), axis=-1), \
            np.concatenate((motion1_feet, motion2_feet), axis=-1), \
            np.concatenate((marker_height_1, marker_height_2), axis=-1), sdf_points"
    batch_name = [item[0] for item in batch]
    batch_text = [item[1] for item in batch]
    batch_motion1 = torch.stack([torch.from_numpy(item[2]) for item in batch], dim=0).type(torch.float32).to(device, non_blocking=True)
    batch_motion2 = torch.stack([torch.from_numpy(item[3]) for item in batch], dim=0).type(torch.float32).to(device, non_blocking=True)
    batch_gt_length = torch.stack([torch.tensor(item[4]) for item in batch], dim=0).long().to(device, non_blocking=True)
    batch_cond_length = torch.stack([torch.tensor(item[5]) for item in batch], dim=0).long().to(device, non_blocking=True)
    batch_motion_R = torch.stack([torch.from_numpy(item[6]) for item in batch], dim=0).type(torch.float32).to(device, non_blocking=True)
    batch_motion_T = torch.stack([torch.from_numpy(item[7]) for item in batch], dim=0).type(torch.float32).to(device, non_blocking=True)
    batch_motion_feet = torch.stack([torch.from_numpy(item[8]) for item in batch], dim=0).long().to(device, non_blocking=True)
    batch_feet_height_thresh = torch.stack([torch.from_numpy(item[9]) for item in batch], dim=0).type(torch.float32).to(device, non_blocking=True)
    # Transfer sdf_points to cuda, because 3D data takes much time in the device transition. (~0.5s, but also renders 512MB for 16*4*128*128*128 cuda tensor)
    batch_sdf_points = torch.stack([item[10] for item in batch], dim=0).type(torch.float32).to(device, non_blocking=True)

    return batch_name, batch_text, batch_motion1, batch_motion2, batch_gt_length, batch_cond_length, \
            batch_motion_R, batch_motion_T, batch_motion_feet, batch_feet_height_thresh, batch_sdf_points



class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, batch_size, num_workers, sdf_points_res):
        """
        Initialize LightningDataModule for ProHMR training
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode containing necessary dataset info.
            dataset_cfg (CfgNode): Dataset configuration file
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sdf_points_res = sdf_points_res

    def setup(self, stage = None):
        """
        Create train and validation datasets
        """
        if self.cfg.NAME == "interhuman":
            self.train_dataset = InterHumanDataset(self.cfg, self.sdf_points_res)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        """
        Return train dataloader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,  # Important for speed up.
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
            # collate_fn=partial(collate_fn, device='cpu')
            )
