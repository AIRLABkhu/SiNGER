import os
import torch
from torchvision import datasets, transforms
from ._common import make_loader
from .imagenet import (
    get_imagenet_train_transform,
    get_imagenet_test_transform,
)

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/inat')
TRAIN_SPLIT_RATIO = 0.8


def make_split(num_samples: int, train_size: float=TRAIN_SPLIT_RATIO, seed: int=1234):
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(num_samples, generator=generator)
    train_size = int(train_size * num_samples)
    tr_indices: torch.Tensor = indices[:train_size].sort().values
    vl_indices: torch.Tensor = indices[train_size:].sort().values
    return tr_indices, vl_indices


class INat2019(datasets.INaturalist):
    def __init__(
        self,
        root: str,
        train: bool,
        transform=None,
        target_transform=None,
    ):
        super().__init__(
            root=root, version='2019', target_type='full',
            transform=transform,
            target_transform=target_transform,
            download=False,
        )
        self.train = train
        split_indices = make_split(len(self.index))
        if train:
            self.index = [self.index[i] for i in split_indices[0]]
        else:
            self.index = [self.index[i] for i in split_indices[1]]
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_inat2019_dataloaders(batch_size, val_batch_size, num_workers, use_ddp, use_subset=False,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], img_size=224, resize_size=256, crop_size=224):
    train_transform = get_imagenet_train_transform(mean, std, img_size=img_size)
    train_set = INat2019(data_folder, train=True, transform=train_transform)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_inat2019_val_loader(val_batch_size, use_ddp, mean, std, resize_size=resize_size, crop_size=crop_size)
    return train_loader, test_loader, num_data

def get_inat2019_dataloaders_sample(batch_size, val_batch_size, num_workers, use_ddp, use_subset=False, k=4096, 
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], img_size=224, resize_size=256, crop_size=224):
    train_transform = get_imagenet_train_transform(mean, std, img_size=img_size)
    train_set = INat2019(data_folder, train=True, transform=train_transform)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_inat2019_val_loader(val_batch_size, use_ddp, mean, std, resize_size=resize_size, crop_size=crop_size)
    return train_loader, test_loader, num_data

def get_inat2019_val_loader(val_batch_size, use_ddp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resize_size=256, crop_size=224):
    test_transform = get_imagenet_test_transform(mean, std, resize_size, crop_size)
    test_set = INat2019(data_folder, train=False, transform=test_transform)
    test_loader = make_loader(test_set, val_batch_size, num_workers=16, shuffle=False, use_ddp=use_ddp)
    return test_loader
