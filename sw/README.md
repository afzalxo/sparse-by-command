# Software: Training and Evaluation Pipeline

This directory contains the PyTorch training pipeline, CARLA evaluation scripts, dataset collection tools, and model definitions for the task-conditional visuomotor driving controller.

## Directory Structure

```
sw/
├── models/
│   └── carla_model.py            # CarlaVLA_Dense and CarlaVLA_Gated model definitions
├── training/
│   ├── train_carla_dense.py      # Phase 1: dense baseline training (200 epochs)
│   └── train_carla_sparse.py     # Phase 2+3: soft/hard mask training (50+50 epochs)
├── evaluation/
│   ├── eval_fpga_emulated.py     # FPGA-emulated evaluation (INT8 quantized on GPU)
│   ├── eval_fpga_routes.py       # Multi-route evaluation with FPGA
│   ├── eval_multi_run.py         # Multi-run evaluation with variance reporting
│   └── eval_static_pruning.py    # Static structured pruning baseline
├── data_collection/
│   ├── collect_dataset_carla.py  # CARLA expert data collection with DAgger
│   └── rebalance_dataset.py      # Command distribution rebalancing (90% straight drop)
├── dataloaders/
│   ├── carla_dataloaders.py      # PyTorch Dataset/DataLoader for CARLA frames
│   └── carla_augmentations.py    # Horizontal flip with command relabelling
└── carla_utils/
    ├── eval_utils.py             # Closed-loop evaluation driver (dense + gated modes)
    ├── route_utils.py            # Route definitions and waypoint management
    ├── traffic_utils.py          # NPC vehicle spawning and traffic management
    └── visualization_utils.py    # Evaluation visualization utilities
```

## Model Architecture

The backbone is an 8-layer stride-2 CNN (output channels: 32, 64, 128, 256, 512, 512, 512, 1024) reducing 256x256x3 to a 1024-dim feature vector. A measurement encoder fuses the visual embedding with previous control state (1152->512, ReLU, 50% dropout). Six parallel branch heads (512->256->256->2, Tanh) output steering and acceleration per driving command.

The tile gating MLP (6->64->182, sigmoid) adds <1,200 parameters and predicts per-tile binary execution masks from the driving command.

## Three-Phase Training Pipeline

### Phase 1: Dense Training
```bash
python training/train_carla_dense.py \
    --epochs 200 --lr 2e-4 --batch-size 256 \
    --dataset-dir /path/to/datasets/carla_dataset_dynamic
```
Trains the dense backbone with branched L1 loss, quadratic steering weighting, and DAgger-style augmentation.

### Phase 2: Soft Mask Training
```bash
python training/train_carla_sparse.py \
    --dense-checkpoint <phase1.pth> \
    --phase 2 --epochs 50 --lambda-sparsity 0.002
```
Attaches the gating network, trains with continuous sigmoid masks under sparsity penalty.

### Phase 3: Hard Mask Training
```bash
python training/train_carla_sparse.py \
    --phase2-checkpoint <phase2.pth> \
    --phase 3 --epochs 50
```
Binarizes masks at threshold 0.5, fine-tunes with straight-through estimator.

## Evaluation

### Static Evaluation (0 NPC vehicles)
```bash
python evaluation/eval_fpga_routes.py \
    --model-path <checkpoint.pth> --gated \
    --n-vehicles 0
```

### Dynamic Evaluation (40 NPC vehicles, 3 runs)
```bash
python evaluation/eval_multi_run.py \
    --model-path <checkpoint.pth> --gated \
    --n-vehicles 40 --n-runs 3
```

### Static Pruning Baseline
```bash
python evaluation/eval_static_pruning.py \
    --model-path <dense_checkpoint.pth> \
    --sparsity-levels 0.2 0.4 0.5 0.6 0.7
```

## Dataset Collection

### Collect Expert Demonstrations
```bash
python data_collection/collect_dataset_carla.py \
    --town Town10HD_Opt --speed 15 \
    --n-episodes 100 --output-dir /path/to/dataset
```
Collects 256x256 RGB frames at 20 FPS with DAgger-style perturbations and expert recovery.

### Rebalance Command Distribution
```bash
python data_collection/rebalance_dataset.py \
    --dataset-dir /path/to/dataset --drop-straight 0.9
```
Drops 90% of straight-driving frames to balance the command distribution.

## Prerequisites

- Python 3.8+
- PyTorch 1.13+ with CUDA
- CARLA 0.9.14 simulator
- Additional packages: `numpy`, `pandas`, `opencv-python`, `wandb` (optional)

## Loss Function

Phase 2 (soft masks):
```
L_soft = L_task(y_hat, y) + lambda * (1/T) * sum(sigma(g_i))
```

Phase 3 (hard masks with STE):
```
m_i = 1[sigma(g_i) > 0.5]
L_hard = L_task(y_hat_masked, y) + lambda * (1/T) * sum(m_i)
```
where T=182 predictable tiles and lambda ramps from 0 to 0.002.
