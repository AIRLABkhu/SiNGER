# Knowledge Distillation for Vision Foundation Models

### Installation

Environments:

- Python 3.11.11
- PyTorch 2.6.0
- torchvision 0.21.0

Install the package:

```
conda env create -f environment.yaml
conda activate distill
```

### Getting started

0. Wandb as the logger

- The registeration: <https://wandb.ai/home>.
- If you don't want wandb as your logger, set `CFG.LOG.WANDB` as `False` at `mdistiller/engine/cfg.py`.

1. DDP setup

    ```bash
    # for instance, FitNet method.
    # get the number of active devices and set the number of OpenMP threads.
    export CUDA_DEVICE_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
    export OMP_NUM_THREADS=4
    ```

2. Training on ImageNet

- Download the dataset at <https://image-net.org/> and put them to `./data/imagenet`

  ```bash
  # for dinov2 training
  torchrun --nproc-per-node=$CUDA_DEVICE_COUNT tools/train.py --cfg configs/imagenet/dinov2/amd-sner.yaml ./configs/imagenet/optim/adamw-dinov2.yaml
  torchrun --nproc-per-node=$CUDA_DEVICE_COUNT tools/train.py --cfg configs/imagenet/vit/amd-sner.yaml ./configs/imagenet/optim/adamw.yaml
  ```

- Config examples for experiment are in `configs/imagenet/dinov2` and `configs/imagenet/vit`. If you want to custom config, modify  `mdistiller/engines/cfg.py` and `mdisitller/disitllers/....py`

  ```yaml
  AMD:
    M_LAYERS: [15] # for distillation layer
    ALIGN_TYPE: 'mse'
    INPUT_SIZE: [518, 518] 
    LOSS:
      FEAT_WEIGHT: 1.0
    SNER: # SNER params. 
      RANK: 16
      OUTLIER_Q: 0.95
      METHOD: 'sner'
  ```


3. Evaluation

- You can evaluate the performance of our models or models trained by yourself.

  ```bash
  # evaluate students 
  # ImageNet-1K classification
  export SET_DC="--nproc-per-node=$CUDA_DEVICE_COUNT"
  export ARGS="--epochs 150 --batch-size 64 --test-batch-size 64 -lr 0.5"
  export SAVE_DIR="dinov2-baselines/amd-sner,dinov2-large,dinov2-tiny,layer15,mse"
  torchrun $SET_DC tools/lineval/imagenet.py $SAVE_DIR $ARGS 
  python tools/lineval/test/imagenet.py $SAVE_DIR

  # NYUd Depth estimation
  export SET_DC="--nproc-per-node=$CUDA_DEVICE_COUNT"
  export ARGS="--epochs 200 --batch-size 64 --test-batch-size 64 -lr 0.5"
  export SAVE_DIR="dinov2-baselines/amd-sner,dinov2-large,dinov2-tiny,layer15,mse"
  torchrun $SET_DC tools/lineval/nyud.py $SAVE_DIR $ARGS 
  python tools/lineval/test/nyud.py $SAVE_DIR

  # ADE-20K Semantic segmentation
  export SET_DC="--nproc-per-node=$CUDA_DEVICE_COUNT"
  export ARGS="--epochs 200 --batch-size 64 --test-batch-size 64 -lr 0.5"
  export SAVE_DIR="dinov2-baselines/amd-sner,dinov2-large,dinov2-tiny,layer15,mse"
  torchrun $SET_DC tools/lineval/ade20k.py $SAVE_DIR $ARGS
  python tools/lineval/test/ade20k.py $SAVE_DIR

  ```

- You can dpt evaluation too. just change `linveal` to `dpteval`
