import os
from argparse import ArgumentParser, Namespace
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from mdistiller.dataset.nyud_v2 import get_nyud_dataloaders

from tools.lineval.utils import init_parser
from tools.lineval.nyud import crop_resize

from tools.dpteval.dpt.models import DPTDepthModel
from tools.dpteval.utils import load_head_checkpoint

def compute_depth_metrics(pred, gt, mask=None, max_depth=10.0, eps=1.0E-8):
    """
    pred, gt: torch.Tensor of shape (B, H, W), in meters
    mask: optional torch.BoolTensor of shape (B, H, W)
    Returns: dict with average metrics over batch
    """
    pred = pred.float()
    gt = gt.float()

    if mask is None:
        mask = (gt > 0) & (gt < max_depth) & (pred > 0) & (pred < max_depth)
    pred_valid = pred[mask]
    gt_valid = gt[mask]

    rmse = torch.sqrt(torch.mean((pred_valid - gt_valid) ** 2))
    mae = torch.mean(torch.abs(pred_valid - gt_valid))
    rel = torch.mean(torch.abs(pred_valid - gt_valid) / (gt_valid + eps))

    ratio = torch.maximum(pred_valid / (gt_valid + eps), gt_valid / (pred_valid + eps))
    delta1 = torch.mean((ratio < 1.25 ** 1).float())
    delta2 = torch.mean((ratio < 1.25 ** 2).float())
    delta3 = torch.mean((ratio < 1.25 ** 3).float())

    return {
        'RMSE': rmse.item(),
        'MAE': mae.item(),
        'REL': rel.item(),
        'δ<1.25': delta1.item(),
        'δ<1.25^2': delta2.item(),
        'δ<1.25^3': delta3.item(),
    }


def main(args: Namespace):
    DEVICE = args.device
    
    # DataLoaders, Models
    _, test_loader, _ = get_nyud_dataloaders(
        1, args.batch_size,
        args.num_workers, use_ddp=False,
    )

    model = DPTDepthModel(
        checkpoint=args.expname,
        non_negative=True,
        enable_attention_hooks=False,
    )
    model = model.cuda(DEVICE)
    
    decoder_state_dict = load_head_checkpoint(
        args.expname, tag=args.tag, 
        when=args.dpteval_when, dataset='nyud',
        eval_tag=args.dpteval_tag
    )['decoder_state_dict']
    
    print(model.load_state_dict(decoder_state_dict, strict=False))
    
    # Test Loop
    with tqdm(test_loader, desc=f' TEST', dynamic_ncols=True) as bar, torch.no_grad():
        total_loss, preds_all, targets_all = 0, [], []
        for input, target in bar:
            input, target = crop_resize(input, target, size=224, random_crop=False)
            pred = model(input.cuda(DEVICE)) # (B, 1, H, W)
            pred = pred.squeeze(1)
            
            batch_size = input.size(0)
            loss = torch.nn.functional.cross_entropy(pred, target.to(DEVICE))
            target_depth = 0.5 + target * 9.5
            pred_depth = 0.5 + pred * 9.5
            
            total_loss += loss.cpu().item() * batch_size
            preds_all.append(pred_depth.cpu())
            targets_all.append(target_depth.cpu())
        
    preds = torch.cat(preds_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    mean_loss = total_loss / len(preds)
    
    metrics = compute_depth_metrics(preds, targets)
    metrics['loss'] = mean_loss
    max_key_len = max(map(len, metrics.keys()))
    for key, val in metrics.items():
        print(f'{key:<{max_key_len}}: {val:.4f}')


if __name__ == '__main__':
    parser = ArgumentParser('dpteval.test.nyud')
    init_parser(parser, defaults=dict(batch_size=16))
    parser.add_argument('--temperature', '-T', type=float, default=0.03)
    parser.add_argument('--dpteval-when', '-dw', type=str, default=None)
    parser.add_argument('--dpteval-tag', '-dt', type=str, default='best')
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        pass
