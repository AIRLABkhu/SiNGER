import os
import torch
from torch.utils.data import Dataset, DataLoader


class FeatureDataset(Dataset):
    def __init__(self, dname: str):
        self.dname = dname
        self.fnames = os.listdir(self.dname)
    
    def __len__(self) -> int:
        return len(self.fnames)
    
    def __getitem__(self, index: int) -> tuple:
        path = os.path.join(self.dname, self.fnames[index])
        f, t = torch.load(path, map_location='cpu', weights_only=False)
        return f, t


def get_feature_loader(dname: str, batch_size: int, num_workers: int, shuffle: bool):
    dataset = FeatureDataset(dname)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )
    return loader
