import os
from argparse import ArgumentParser, Namespace
from tqdm.auto import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from torch.utils.data import (
    TensorDataset, DataLoader, 
)

from mdistiller.dataset.imagenet import get_imagenet_dataloaders
from mdistiller.models._base import ModelBase
from mdistiller.models import imagenet_model_dict

from tools.lineval.utils import (
    init_parser,
    prepare_lineval_dir,
    load_from_checkpoint,
)
from tools.lineval.imagenet import ClsHead
from tools.lineval.extract.utils import get_feature_loader


def main(args: Namespace):
    DEVICE = args.device
    EPOCHS = args.epochs
    
    _, log_filename, best_filename, last_filename = prepare_lineval_dir(
        args.expname, 
        tag=str(args.tag), 
        mode='extract',
        dataset='imagenet1k', 
        args=vars(args),
        use_nowstr=False,
    )
    dname_tr = os.path.join(os.path.dirname(log_filename), 'dump_tr')
    dname_te = os.path.join(os.path.dirname(log_filename), 'dump_te')
    
    # DataLoaders, Models
    resize_size = int(args.img_size * 256 / 224)
    crop_size = args.img_size
    train_loader, test_loader, _ = get_imagenet_dataloaders(
        args.batch_size, args.test_batch_size,
        args.num_workers, use_ddp=False, img_size=args.img_size,
        resize_size=resize_size, crop_size=crop_size
    )
    if args.timm_model is not None:
        print(f'Loading {args.timm_model} from timm')
        model = imagenet_model_dict[args.timm_model](pretrained=True)
    else:
        model, _ = load_from_checkpoint(args.expname, tag=args.tag)
    model: ModelBase = model.cuda(DEVICE)
    model.set_input_size(args.img_size)
    head = ClsHead(model.embed_dim, take_cls_only=True).cuda(DEVICE)
    
    optimizer = optim.SGD(
        head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=EPOCHS*len(train_loader),
        eta_min=1.0E-8,
    )
    
    # Extract Loop
    if os.path.exists(dname_tr):
        print('Use dumps for train set.')
    else:
        os.makedirs(dname_tr, exist_ok=True)
        with torch.no_grad():
            sidx = 0
            for input, target, _ in tqdm(train_loader, desc='TRAIN', dynamic_ncols=True):
                feature = model.forward(input.cuda(DEVICE))[1]['feats'][-1].cpu()
                feature = feature[:, 0]
                for f, t in zip(feature, target):
                    torch.save((f, t), os.path.join(dname_tr, f'{sidx}.pt'))
                    sidx += 1
    
    if os.path.exists(dname_te):
        print('Use dumps for test set.')
    else:
        os.makedirs(dname_te, exist_ok=True)
        with torch.no_grad():
            for input, target in tqdm(test_loader, desc='TEST', dynamic_ncols=True):
                feature = model.forward(input.cuda(DEVICE))[1]['feats'][-1].cpu()
                feature = feature[:, 0]
                for f, t in zip(feature, target):
                    torch.save((f, t), os.path.join(dname_te, f'{sidx}.pt'))
                    sidx += 1
    
    train_loader = get_feature_loader(dname_tr, batch_size=args.batch_size, num_workers=4, shuffle=True)
    test_loader = get_feature_loader(dname_te, batch_size=args.test_batch_size, num_workers=4, shuffle=False)
    
    # Training Loop
    best_top1 = -torch.inf
    train_loss_list, test_loss_list = [], []
    train_top1_list, test_top1_list = [], []
    train_top5_list, test_top5_list = [], []
    for epoch in range(args.epochs):
        
        with tqdm(train_loader, desc=f'TRAIN {epoch+1}', dynamic_ncols=True) as bar:
            total_loss, correct_top1, correct_top5, total = 0, 0, 0, 0
            for x, target in bar:
                pred_logit = head(x.to(DEVICE)) # (B, 1000)

                loss = F.cross_entropy(pred_logit, target.to(DEVICE))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                with torch.no_grad():
                    loss = F.cross_entropy(pred_logit, target.to(DEVICE), reduction='none')
                    
                    i_top5_all = pred_logit.topk(k=5, dim=1).indices
                    target_all = target.to(DEVICE).unsqueeze(-1)
                    
                    total_loss += loss.sum().cpu().item()
                    correct_top1 += (i_top5_all[:, 0:1] == target_all).sum().cpu().item()
                    correct_top5 += (i_top5_all[:, 0:5] == target_all).sum().cpu().item()
                    total += loss.size(0)
                    
                    bar.set_postfix(dict(
                        lr=optimizer.param_groups[0]['lr'],
                        loss=total_loss/total,
                        top1=correct_top1/total*100.0,
                        top5=correct_top5/total*100.0,
                    ))
            train_top1 = correct_top1 / total
            train_top5 = correct_top5 / total
            train_loss = total_loss / total

        with tqdm(test_loader, desc=f' TEST {epoch+1}', dynamic_ncols=True) as bar, torch.no_grad():
            total_loss, correct_top1, correct_top5, total = 0, 0, 0, 0
            for x, target in bar:
                pred_logit = head(x.to(DEVICE)) # (B, 1000)
                
                loss = F.cross_entropy(pred_logit, target.to(DEVICE), reduction='none', ignore_index=-1)
                    
                i_top5_all = pred_logit.topk(k=5, dim=1).indices
                target_all = target.to(DEVICE).unsqueeze(-1)
                
                total_loss += loss.sum().cpu().item()
                correct_top1 += (i_top5_all[:, 0:1] == target_all).sum().cpu().item()
                correct_top5 += (i_top5_all[:, 0:5] == target_all).sum().cpu().item()
                total += loss.size(0)
                
                bar.set_postfix(dict(
                    loss=total_loss/total,
                    top1=correct_top1/total*100.0,
                    top5=correct_top5/total*100.0,
                ))
            test_top1 = correct_top1 / total
            test_top5 = correct_top5 / total
            test_loss = total_loss / total
        
        # Logging
        train_loss_list.append(train_loss)
        train_top1_list.append(train_top1)
        train_top5_list.append(train_top5)
        test_loss_list.append(test_loss)
        test_top1_list.append(test_top1)
        test_top5_list.append(test_top5)
        
        with open(log_filename, 'a') as file:
            print(f'- epoch: {epoch+1}', file=file)
            print(f'  train_loss: {train_loss:.4f}', file=file)
            print(f'  train_top1: {train_top1*100.0:.2f}%', file=file)
            print(f'  train_top5: {train_top5*100.0:.2f}%', file=file)
            print(f'  test_loss: {test_loss:.4f}', file=file)
            print(f'  test_top1: {test_top1*100.0:.2f}%', file=file)
            print(f'  test_top5: {test_top5*100.0:.2f}%', file=file)
            print(file=file)
        
        ckpt = dict(
            epoch=epoch+1,
            train_loss=train_loss_list,
            train_top1=train_top1_list,
            train_top5=train_top5_list,
            test_loss=test_loss_list,
            test_top1=test_top1_list,
            test_top5=test_top5_list,
            head={
                key: val.clone().detach().cpu()
                for key, val in head.state_dict().items()
            },
        )
        if test_top1 > best_top1:
            best_top1 = test_top1
            torch.save(ckpt, str(best_filename))
        torch.save(ckpt, str(last_filename))


if __name__ == '__main__':
    parser = ArgumentParser('lineval.extract.imagenet1k')
    init_parser(parser, defaults=dict(epochs=1000))
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    finally:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
