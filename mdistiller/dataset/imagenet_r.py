import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from ._common import make_loader
from .imagenet import (
    ImageNet,
    get_imagenet_train_transform,
    get_imagenet_test_transform,
)


data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/imagenet')
data_folder_r = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/imagenet-r')


def make_r_to_1k_map():
    in_1k_cls = sorted(os.listdir(os.path.join(data_folder, 'train')))
    in_r_cls = sorted(os.listdir(data_folder_r))

    cursor_1k, cursor_r = 0, 0
    r_to_1k_map = []
    while len(r_to_1k_map) < len(in_r_cls):
        if in_1k_cls[cursor_1k] == in_r_cls[cursor_r]:
            r_to_1k_map.append(cursor_1k)
            cursor_r += 1
        cursor_1k += 1
    return r_to_1k_map


R_TO_1K_CLASS_MAP = make_r_to_1k_map()


class ImageNetR(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, R_TO_1K_CLASS_MAP[target], index


def get_imagenet_r_dataloaders(batch_size, val_batch_size, num_workers, use_ddp, use_subset=False,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], img_size=224, resize_size=256, crop_size=224):
    train_transform = get_imagenet_train_transform(mean, std, img_size=img_size)
    train_folder = os.path.join(data_folder, 'train')
    train_set = ImageNet(train_folder, transform=train_transform)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_imagenet_r_val_loader(val_batch_size, use_ddp, mean, std, resize_size=resize_size, crop_size=crop_size)
    return train_loader, test_loader, num_data

def get_imagenet_r_val_loader(val_batch_size, use_ddp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resize_size=256, crop_size=224):
    test_transform = get_imagenet_test_transform(mean, std, resize_size, crop_size)
    test_folder = data_folder_r
    test_set = ImageNetR(test_folder, transform=test_transform)
    test_loader = make_loader(test_set, val_batch_size, num_workers=16, shuffle=False, use_ddp=use_ddp)
    return test_loader
