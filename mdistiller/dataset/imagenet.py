import os
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from ._common import make_loader


data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/imagenet')
subset_indices_1000_5 = torch.load(os.path.join(data_folder, 'subset-1000-5.pt'), weights_only=False)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def denormalize(img: torch.Tensor):
    if img.ndim == 3:
        dim3 = True
        img = img[None]
    elif img.ndim == 4:
        dim3 = False
    else:
        raise RuntimeError(img.ndim)
    
    if img.size(-1) == 3:
        img = img.permute(0, 3, 1, 2)
        channel_last = True
    elif img.size(1) == 3:
        channel_last = False
    else:
        raise RuntimeError(img.shape)
    
    dtype, device = img.dtype, img.device
    mean = torch.tensor(MEAN).reshape(1, 3, 1, 1).to(dtype=dtype, device=device)
    std = torch.tensor(STD).reshape(1, 3, 1, 1).to(dtype=dtype, device=device)
    
    img_denormed = img * std + mean
    if channel_last:
        img_denormed = img_denormed.permute(0, 2, 3, 1)
    if dim3:
        img_denormed = img_denormed[0]
    return img_denormed


class ImageNet(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class ImageNetInstanceSample(ImageNet):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, folder, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(folder, transform=transform)

        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.samples[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target, index = super().__getitem__(index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index

def get_imagenet_train_transform(mean, std, img_size=224):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform

def get_imagenet_test_transform(mean, std, resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return test_transform

def get_imagenet_dataloaders(batch_size, val_batch_size, num_workers, use_ddp, use_subset=False,
    mean=MEAN, std=STD, img_size=224, resize_size=256, crop_size=224):
    train_transform = get_imagenet_train_transform(mean, std, img_size=img_size)
    train_folder = os.path.join(data_folder, 'train')
    train_set = ImageNet(train_folder, transform=train_transform)
    if use_subset:
        train_set = Subset(train_set, subset_indices_1000_5)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_imagenet_val_loader(val_batch_size, use_ddp, mean, std, resize_size=resize_size, crop_size=crop_size)
    return train_loader, test_loader, num_data

def get_imagenet_dataloaders_sample(batch_size, val_batch_size, num_workers, use_ddp, use_subset=False, k=4096, 
    mean=MEAN, std=STD, img_size=224, resize_size=256, crop_size=224):
    train_transform = get_imagenet_train_transform(mean, std, img_size=img_size)
    train_folder = os.path.join(data_folder, 'train')
    train_set = ImageNetInstanceSample(train_folder, transform=train_transform, is_sample=True, k=k)
    if use_subset:
        train_set = Subset(train_set, subset_indices_1000_5)
    num_data = len(train_set)
    train_loader = make_loader(train_set, batch_size, num_workers, shuffle=True, use_ddp=use_ddp)
    test_loader = get_imagenet_val_loader(val_batch_size, use_ddp, mean, std, resize_size=resize_size, crop_size=crop_size)
    return train_loader, test_loader, num_data

def get_imagenet_val_loader(val_batch_size, use_ddp, mean=MEAN, std=STD, resize_size=256, crop_size=224):
    test_transform = get_imagenet_test_transform(mean, std, resize_size, crop_size)
    test_folder = os.path.join(data_folder, 'val')
    test_set = ImageFolder(test_folder, transform=test_transform)
    test_loader = make_loader(test_set, val_batch_size, num_workers=16, shuffle=False, use_ddp=use_ddp)
    return test_loader
