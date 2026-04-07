# Compiler: SW-to-HW Deployment Pipeline

This directory contains the offline compiler that bridges the gap between a trained PyTorch model and the FPGA accelerator's INT8 tiled datapath. It performs quantization, data packing, and NISA instruction stream generation.

## Files

```
compiler/
├── extract_data_for_fpga.py      # Main deployment compiler
└── extract_task_circuit.py       # Task-conditional mask extraction
```

## Pipeline Stages

### 1. Quantization (`extract_data_for_fpga.py`)
- Runs 300 calibration samples through the model to record per-layer maximum activations
- Computes quantization shifts: `shift = ceil(log2(act_max * cumulative_scale / Q_max))` where Q_max = 127
- Quantizes weights to INT8 (per-tensor symmetric scaling)
- Quantizes biases to INT32, clamped to 28-bit accumulator range

### 2. Data Packing
- Feature maps: `[C, H, W]` -> `[H_tiles, C_slots, H_local, W, C_local]` (IC_PAR=16 alignment)
- Weights: `[OC, IC, Kh, Kw]` -> `[OC_tiles, IC_tiles, Ky, Kx, OC_par, IC_par]`
- Zero-pads channel dimensions to parallelism boundaries

### 3. Memory Layout
- Contiguous HBM region: 64 KB instructions, packed input image, all layers' weights and biases
- Per-layer byte offsets recorded in NISA instructions

### 4. Instruction Generation
- One 64-byte NISA instruction per layer
- Ping-pong bank selection alternates across layers
- Tile masks encoded from gating network output

### 5. Task Mask Extraction (`extract_task_circuit.py`)
- Evaluates the gating network for each of the 6 driving commands
- Binarizes sigmoid outputs at threshold 0.5
- Writes per-task tile masks for model spec files

## Usage

### Full Deployment (Dense)
```bash
python extract_data_for_fpga.py \
    --model-path <checkpoint.pth> \
    --dataset-dir /path/to/calibration_dataset \
    --output-dir fpga_data \
    --n-calibration-samples 300
```

### Extract Task Masks (Sparse)
```bash
python extract_task_circuit.py \
    --model-path <gated_checkpoint.pth> \
    --output-dir fpga_data
```

## Output Files

```
fpga_data/
├── model_spec.txt           # Layer config (dims, shifts, masks, file paths)
├── task_masks.txt            # Per-task tile masks (6 rows x 8 layers)
├── input_vision.bin          # Packed input image (calibration sample)
├── extra_features.bin        # Measurement embedding input
├── weights_conv0.bin ...     # Per-layer quantized weights
├── bias_conv0.bin ...        # Per-layer quantized biases
└── golden_conv0.bin ...      # Per-layer golden reference outputs (INT8)
```

## Model Specification Format

Each line in `model_spec.txt` defines one layer:
```
[name] [w h ic oc] [stride] [quant_shift] [flags] [bank] [sparse] [oc_mask_lo] [oc_mask_hi] [weight_file] [bias_file] [golden_file]
```
