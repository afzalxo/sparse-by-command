# Sparse by Command: Task-Conditional Compute Skipping for Multi-Task Inference Accelerators

This repository contains the full HW/SW co-design stack for task-conditional tile sparsity on FPGA inference accelerators. A lightweight gating network predicts per-tile binary execution masks from a task command, and the accelerator's instruction set encodes these masks to skip entire output channel tiles at zero overhead.

## Repository Structure

```
.
├── hw/                    # FPGA accelerator (RTL, host program, build scripts)
│   ├── rtl/               # SystemVerilog/Verilog source files
│   ├── host/              # OpenCL C++ host application
│   └── build_scripts/     # Vivado/Vitis build automation
├── sw/                    # Software training and evaluation pipeline
│   ├── models/            # PyTorch model definitions
│   ├── training/          # Dense and sparse training scripts
│   ├── evaluation/        # CARLA closed-loop evaluation
│   ├── data_collection/   # CARLA dataset collection
│   ├── dataloaders/       # PyTorch dataloaders and augmentation
│   └── carla_utils/       # CARLA server, traffic, and route utilities
├── compiler/              # SW-to-HW deployment pipeline
│   ├── extract_data_for_fpga.py      # INT8 quantization and data packing
│   └── extract_task_circuit.py       # Task-conditional mask extraction
└── README.md
```

## System Overview

The system co-designs three layers:

1. **Task-Conditional Gating Network** (`sw/models/`): A lightweight MLP (<0.12% of backbone parameters) that maps a one-hot task command vector to a 182-bit binary tile mask, trained jointly with the backbone through a three-phase pipeline (dense, soft masks, hard masks).

2. **Neural Instruction Set Architecture** (`hw/rtl/instruction_scheduler.sv`): Each 64-byte NISA instruction carries two 64-bit tile mask fields. The instruction scheduler delivers these to the tile manager, which skips masked tiles with zero overhead.

3. **Tiled Inference Accelerator** (`hw/rtl/`): A configurable INT8 accelerator with PP_PAR=8, IC_PAR=16, OC_PAR=16 (2,048 MACs/cycle), double-buffered memory, and cycle-accurate performance counters. Deployed on AMD/Xilinx Alveo U50 with HBM2 at 287 MHz.

## Key Results

- **FLOP Reduction**: 66-76% per task (task-conditional tile masking)
- **Latency Reduction**: 51-59% (9.12 ms dense to 3.74-4.44 ms sparse)
- **Speedup**: 2.1-2.4x on-device
- **Energy**: 108-128 uJ per sparse inference (vs 263 uJ dense)
- **Driving Quality**: 100% route completion on three hard CARLA evaluation routes
- **Resource Utilization**: 27% LUTs, 35% DSPs on Alveo U50

## Prerequisites

### Hardware Build
- AMD/Xilinx Vitis 2022.2 (or compatible)
- AMD/Xilinx XRT runtime
- Target platform: `xilinx_u50_gen3x16_xdma_5_202210_1`

### Software Training
- Python 3.8+
- PyTorch 1.13+
- CARLA 0.9.14 (UE4) or 0.10.0 (UE5)
- wandb (optional, for experiment tracking)

## Quick Start

### 1. Train the Dense Model
```bash
cd sw/training
python train_carla_dense.py --epochs 200 --lr 2e-4 --batch-size 256
```

### 2. Train with Task-Conditional Sparsity
```bash
python train_carla_sparse.py --dense-checkpoint <path> --phase2-epochs 50 --phase3-epochs 50 --lambda-sparsity 0.002
```

### 3. Extract Data for FPGA
```bash
cd compiler
python extract_data_for_fpga.py --model-path <checkpoint> --dataset-dir <dataset> --output-dir fpga_data
```

### 4. Build FPGA Bitstream
```bash
cd hw/build_scripts
./build_xo_vla.sh        # Package RTL as .xo
./gen_xclbin.sh           # Link and place-and-route to .xclbin
```

### 5. Build and Run Host Application
```bash
cd hw/host
python make_host.py
./app_main.exe <xclbin_path> model_spec.txt
```

See individual subdirectory READMEs for detailed instructions.

## Closed-loop FPGA evaluation - FPGA inference 256x256 - 4x Speed

| StaticEval | DynamicEval (40 NPC) |
| :---: | :---: |
| <video src="https://github.com/user-attachments/assets/0bf6d5d8-66a6-4a70-bd23-65044bdc147f" width="512" controls></video> | <video src="https://github.com/user-attachments/assets/35d21e13-1176-4291-927d-5444b1b5d3a4" width="512" controls></video> |

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{sparsebycommand2026,
  title={Sparse by Command: Task-Conditional Compute Skipping for Multi-Task Inference Accelerators},
  author={Anonymous},
  booktitle={Proceedings of the 59th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year={2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
