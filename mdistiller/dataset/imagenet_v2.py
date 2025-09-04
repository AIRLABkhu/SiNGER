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
data_folder_v2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/imagenet-v2')


class ImageNetv2(ImageFolder):
    ''' https://github.com/modestyachts/ImageNetV2/issues/10#issuecomment-2140713192 '''
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
        return classes, class_to_idx
    
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_imagenet_v2_dataloaders(batch_size, val_batch_size, num_workers, use_ddp, use_subset=False,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], img_size=224, resize_size=256, crop_size=224):
    train_transform = get_imagenet_train_transform(mean, std, img_size=img_size)
    train_folder = os.path.join(data_folder, 'train')
    train_set = ImageNet(train_folder, transform=train_transform)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_imagenet_v2_val_loader(val_batch_size, use_ddp, mean, std, resize_size=resize_size, crop_size=crop_size)
    return train_loader, test_loader, num_data

def get_imagenet_v2_val_loader(val_batch_size, use_ddp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resize_size=256, crop_size=224):
    test_transform = get_imagenet_test_transform(mean, std, resize_size, crop_size)
    test_folder = data_folder_v2
    test_set = ImageNetv2(test_folder, transform=test_transform)
    test_loader = make_loader(test_set, val_batch_size, num_workers=16, shuffle=False, use_ddp=use_ddp)
    return test_loader
