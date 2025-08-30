from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample
from .tiny_imagenet import get_tinyimagenet_dataloader, get_tinyimagenet_dataloader_sample

from .fgvc_aircraft import get_fgvc_aircraft_dataloaders
from .flowers102 import get_flowers102_dataloaders
from .food101 import get_food101_dataloaders
from .imagenet_r import get_imagenet_r_dataloaders
from .inat2019 import get_inat2019_dataloaders
from .oxford_iiit_pet import get_oxford_iiit_pet_dataloaders


def get_dataset(cfg):
    use_ddp = cfg.EXPERIMENT.DDP
    use_subset = cfg.DATASET.SUBSET
    
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
                use_ddp=use_ddp, use_subset=use_subset,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                use_ddp=use_ddp, use_subset=use_subset,
            )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                use_ddp=use_ddp, use_subset=use_subset,
            )
        else:
            resize_ratio = (256/224)
            resize_size = int(round(cfg.DATASET.INPUT_SIZE[0] * resize_ratio))
            train_loader, val_loader, num_data = get_imagenet_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                use_ddp=use_ddp, use_subset=use_subset,
                img_size=cfg.DATASET.INPUT_SIZE, resize_size=resize_size, crop_size=cfg.DATASET.INPUT_SIZE,
            )
        num_classes = 1000
    elif cfg.DATASET.TYPE == "tiny_imagenet":
        if cfg.DISTILLER.TYPE in ("CRD", "CRDKD"):
            train_loader, val_loader, num_data = get_tinyimagenet_dataloader_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                use_ddp=use_ddp, use_subset=use_subset,
            )
        else:
            train_loader, val_loader, num_data = get_tinyimagenet_dataloader(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                use_ddp=use_ddp, use_subset=use_subset,
            )
        num_classes = 200
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes
