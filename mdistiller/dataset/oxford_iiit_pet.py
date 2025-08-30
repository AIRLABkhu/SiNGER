import os
from torchvision import datasets, transforms
from ._common import make_loader
from .imagenet import (
    get_imagenet_train_transform,
    get_imagenet_test_transform,
)

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')


def get_oxford_iiit_pet_dataloaders(batch_size, val_batch_size, num_workers, use_ddp, use_subset=False,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], img_size=224, resize_size=256, crop_size=224):
    train_transform = get_imagenet_train_transform(mean, std, img_size=img_size)
    train_set = datasets.OxfordIIITPet(data_folder, split='trainval', transform=train_transform)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_oxford_iiit_pet_val_loader(val_batch_size, use_ddp, mean, std, resize_size=resize_size, crop_size=crop_size)
    return train_loader, test_loader, num_data

def get_oxford_iiit_pet_dataloaders_sample(batch_size, val_batch_size, num_workers, use_ddp, use_subset=False, k=4096, 
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], img_size=224, resize_size=256, crop_size=224):
    train_transform = get_imagenet_train_transform(mean, std, img_size=img_size)
    train_set = datasets.OxfordIIITPet(data_folder, split='trainval', transform=train_transform)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_oxford_iiit_pet_val_loader(val_batch_size, use_ddp, mean, std, resize_size=resize_size, crop_size=crop_size)
    return train_loader, test_loader, num_data

def get_oxford_iiit_pet_val_loader(val_batch_size, use_ddp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resize_size=256, crop_size=224):
    test_transform = get_imagenet_test_transform(mean, std, resize_size, crop_size)
    test_set = datasets.OxfordIIITPet(data_folder, split='test', transform=test_transform)
    test_loader = make_loader(test_set, val_batch_size, num_workers=16, shuffle=False, use_ddp=use_ddp)
    return test_loader
