import os
import torch
import tqdm
from torchvision import datasets, transforms
from HHInter.datasets.interhuman import InterHumanDataset
from HHInter.configs import get_config
import random
import numpy as np


def calculate_mean_std(dataloder):
    data_collect = []

    # Traversing the dataset calculates the mean and std
    for data in tqdm.tqdm(dataloder, total=len(dataloder)):
        motion1 = data[2]
        motion2 = data[3]
        length = data[4] + data[5]
        for i in range(motion1.shape[0]):
            data_collect.append(motion1[i, :length[i]].view(-1, 204))
            data_collect.append(motion2[i, :length[i]].view(-1, 204))

    data = torch.concatenate(data_collect, dim=0).view(-1, 204)
    mean = torch.mean(data, dim=(0,))
    std = torch.std(data, dim=(0,))

    return mean, std


if __name__ == "__main__":
    data_cfg = get_config("configs/datasets.yaml").interhuman
    dataset = InterHumanDataset(data_cfg, 128)
    # dataloader
    dataloder = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=8,
        drop_last=False,
        persistent_workers=True,
        pin_memory=True
    )

    mean_list = []
    std_list = []
    # calculate mean and std
    for i in range(10):
        mean, std = calculate_mean_std(dataloder)
        print("Mean:", mean[:10])
        print("Std:", std[:10])
        mean_list.append(mean)
        std_list.append(std)
    mean = torch.stack(mean_list, dim=0)
    std = torch.stack(std_list, dim=0)
    print("Mean:", torch.mean(mean, dim=0))
    print("Std:", torch.mean(std, dim=0))
    # save
    torch.save({"mean": torch.mean(mean, dim=0), "std": torch.mean(std, dim=0)}, "mean_std_204.pth")
