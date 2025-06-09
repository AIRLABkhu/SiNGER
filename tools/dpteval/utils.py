import os
import torch
from pathlib import Path
from typing import Literal
from datetime import datetime

def prepare_dir(
    exp_name: str,
    dir_name: str = 'dpteval',
    tag: Literal['latest', 'best']|int='latest',
    dataset: str='imagenet',
    args: dict|None=None,
):
    lineval_dir = Path('output').joinpath(exp_name, dir_name)
    nowstr = datetime.now().strftime('_%y%m%d_%H%M%S')
    log_dir = lineval_dir.joinpath(str(tag), dataset + nowstr)
    log_dir.mkdir(parents=True)
    
    if args is not None:
        cfg_filename = log_dir.joinpath('_cfg.yaml')
        with open(cfg_filename, 'w') as file:
            for key, val in args.items():
                print(f'{key}: {val}', file=file)
    
    log_filename = log_dir.joinpath('log.yaml')
    best_filename = log_dir.joinpath('best.pt')
    last_filename = log_dir.joinpath('last.pt')
    return log_dir, log_filename, best_filename, last_filename

def load_head_checkpoint(
    exp_name: str,
    dir_name: str = 'dpteval',
    tag: Literal['latest', 'best']|int='latest',
    dataset: Literal['imagenet', 'nyud', 'ade20k']='nyud',
    when: str|None=None,
    eval_tag: Literal['last', 'best']|int='last',
):
    filename = os.path.join('output', exp_name, dir_name, str(tag))
    if when is None:
        dirname = sorted(filter(lambda x: x.startswith(f'{dataset}_'), os.listdir(filename)))[-1]
        filename = os.path.join(filename, dirname)
    else:
        filename = os.path.join(filename, f'{dataset}_{when}')
    filename = os.path.join(filename, f'{eval_tag}.pt')
    return torch.load(filename, weights_only=False, map_location='cpu')