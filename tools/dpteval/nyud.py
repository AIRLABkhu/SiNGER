import os
from argparse import ArgumentParser, Namespace
from tqdm.auto import tqdm
import numpy as np

import einops
from typing import Literal
from datetime import datetime
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch import distributed as dist

from mdistiller.dataset.nyud_v2 import get_nyud_dataloaders
from mdistiller.utils import dist_fn

from tools.lineval.utils import init_parser
from tools.lineval.nyud import crop_resize

from tools.dpteval.dpt.models import DPTDepthModel
from tools.dpteval.utils import prepare_dir


def main(args: Namespace):
    rank = int(os.environ['LOCAL_RANK'])
    IS_MASTER = bool(int(os.environ['IS_MASTER_NODE']))
    DEVICE = rank
    EPOCHS = args.epochs
    
    if IS_MASTER:
        _, log_filename, best_filename, last_filename = prepare_dir(
            args.expname, 
            tag='latest', 
            dataset='nyud', 
            args=vars(args)
        )
    
    # DataLoaders, Models
    train_loader, test_loader, _ = get_nyud_dataloaders(
        args.batch_size//world_size, args.test_batch_size//world_size,
        args.num_workers, use_ddp=True,
    )
    
    model = DPTDepthModel(
        checkpoint=args.expname,
        non_negative=True,
        enable_attention_hooks=False,
    )
    model.train()
    model = model.cuda(DEVICE)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    learnable_params = model.module.get_decoder_named_parameters()
    param_names, learnable_params = [*zip(*learnable_params)]
    
    optimizer = optim.Adam(
        learnable_params, 
        lr=args.learning_rate,
    ) # learning rate of 1e-4 is used in dpt.

    # Training Loop
    best_rmse = torch.inf
    train_loss_list, train_rmse_list, test_loss_list, test_rmse_list = [], [], [], []
    for epoch in range(args.epochs):
        with tqdm(train_loader, desc=f'TRAIN {epoch+1}', dynamic_ncols=True, disable=not IS_MASTER) as bar:
            total_loss, total_rmse, total = 0, 0, 0
            for bidx, (input, target) in enumerate(bar):
                input, target = crop_resize(input, target, size=224, random_crop=True)
                pred = model(input.cuda(DEVICE))

                loss = F.mse_loss(pred, target.to(DEVICE))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                with torch.no_grad():
                    target_depth = 0.5 + target * 9.5  # for real depth from normalized target min 0.5m max 10m
                    pred_depth = 0.5 + pred * 9.5
                    
                    loss = F.mse_loss(pred, target.to(DEVICE), reduction='none')

                    loss_all = dist_fn.gather(loss)
                    local_rmse = torch.square(pred_depth - target_depth.to(DEVICE)).flatten(1).mean(dim=1).sqrt()
                    rmse_all = dist_fn.gather(local_rmse)
                    
                    total_loss += loss_all.mean().cpu().item()
                    total_rmse += rmse_all.sum().cpu().item()
                    total += loss_all.size(0)
                    
                    if IS_MASTER:
                        bar.set_postfix(dict(
                            lr=optimizer.param_groups[0]['lr'],
                            loss=total_loss/total,
                            rmse=total_rmse/total,
                        ))
            train_rmse = total_rmse / total
            train_loss = total_loss / total

        with tqdm(test_loader, desc=f' TEST {epoch+1}', dynamic_ncols=True, disable=not IS_MASTER) as bar, torch.no_grad():
            total_loss, total_rmse, total = 0, 0, 0
            for input, target in bar:
                input, target = crop_resize(input, target, size=224, random_crop=False)
                pred = model(input.cuda(DEVICE))
                pred = pred.squeeze(1)

                loss = F.mse_loss(pred, target.to(DEVICE), reduction='none')
                
                loss_all = dist_fn.gather(loss)
                target_depth = 0.5 + target * 9.5
                pred_depth = 0.5 + pred * 9.5
                local_rmse = torch.square(pred_depth - target_depth.to(DEVICE)).flatten(1).mean(dim=1).sqrt()
                rmse_all = dist_fn.gather(local_rmse)
                
                total_loss += loss_all.mean().cpu().item()
                total_rmse += rmse_all.sum().cpu().item()
                total += loss_all.size(0)
                
                if IS_MASTER:
                    bar.set_postfix(dict(
                        loss=total_loss/total,
                        rmse=total_rmse/total,
                    ))
            test_rmse = total_rmse / total
            test_loss = total_loss / total
        
        # Logging
        train_loss_list.append(train_loss)
        train_rmse_list.append(train_rmse)
        test_loss_list.append(test_loss)
        test_rmse_list.append(test_rmse)
        
        if IS_MASTER:
            with open(log_filename, 'a') as file:
                print(f'- epoch: {epoch+1}', file=file)
                print(f'  train_loss: {train_loss:.4f}', file=file)
                print(f'  train_rmse: {train_rmse:.4f}', file=file)
                print(f'  test_loss: {test_loss:.4f}', file=file)
                print(f'  test_rmse: {test_rmse:.4f}', file=file)
                print(file=file)

            ckpt = dict(
                epoch=epoch+1,
                train_loss=train_loss_list,
                train_rmse=train_rmse_list,
                test_loss=test_loss_list,
                test_rmse=test_rmse_list,
                decoder_state_dict={
                    name: param.clone().detach().cpu()
                    for name, param in zip(param_names, learnable_params)
                }
            )
            if test_rmse < best_rmse:
                best_rmse = test_rmse
                torch.save(ckpt, str(best_filename))
            torch.save(ckpt, str(last_filename))

if __name__ == '__main__':
    parser = ArgumentParser('dpteval.nyud')
    init_parser(parser, defaults=dict(epochs=1000))
    args = parser.parse_args()
    
    rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['IS_MASTER_NODE'] = str(int(rank == 0))
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    finally:
        dist.destroy_process_group()