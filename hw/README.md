# Hardware: FPGA Inference Accelerator

This directory contains the complete FPGA accelerator design: SystemVerilog RTL, the OpenCL C++ host application, and Vivado/Vitis build scripts.

## Directory Structure

```
hw/
├── rtl/                  # Synthesizable RTL sources
│   ├── vla_accel_top.sv          # Top-level AXI kernel wrapper
│   ├── instruction_scheduler.sv  # NISA instruction fetch and decode
│   ├── tile_manager.sv           # Central orchestrator (tile iteration, mask checking)
│   ├── conv_accelerator.sv       # Convolution compute pipeline
│   ├── vector_compute_unit.sv    # PP_PAR pixel-parallel compute array
│   ├── processing_element.sv     # Single output-channel PE
│   ├── mac_lane.sv               # INT8 multiply-accumulate datapath
│   ├── conv_controller.sv        # Convolution sequencing FSM
│   ├── line_buffer.sv            # 3-row activation line buffer
│   ├── window_sequencer.sv       # 3x3 sliding window generator
│   ├── result_accumulator.sv     # 28-bit partial sum accumulator
│   ├── tiled_dma.sv              # Input feature map DMA (HBM->URAM)
│   ├── weights_dma.sv            # Weight DMA (HBM->BRAM)
│   ├── bias_dma.sv               # Bias DMA (HBM->registers)
│   ├── output_dma.sv             # Output DMA with quantization (URAM->HBM)
│   ├── gap_unit.sv               # Global average pooling
│   ├── flatten_unit.sv           # Tensor reshape for FC layers
│   ├── memcpy.sv                 # Buffer-to-buffer copy
│   ├── ram_sdp.sv                # Simple dual-port RAM primitive
│   ├── burst_axi_read_master.v   # AXI4 burst read master
│   ├── burst_axi_write_master.v  # AXI4 burst write master
│   └── vla_accel_control_s_axi.v # AXI4-Lite control register bank
├── host/                 # Host application
│   ├── main_carla.cpp            # Single-frame inference with perf counters
│   ├── main_carla_continuous.cpp # Continuous inference server
│   ├── main_jitter_measure.cpp   # Latency benchmarking
│   ├── make_host.py              # Build script (g++ with XRT/OpenCL)
│   ├── xcl2.cpp / xcl2.hpp       # Xilinx OpenCL/XRT utilities
│   ├── xrt.ini                   # XRT runtime configuration
│   └── srcs/
│       ├── globals.hpp           # Constants, NISA_Instruction struct, opcodes
│       ├── packing.hpp           # Data layout conversion (CHW -> tiled)
│       ├── utils.hpp / utils.cpp # Binary I/O, model spec parsing
│       └── verify_output.hpp/cpp # Golden reference verification
└── build_scripts/        # FPGA build automation
    ├── build_xo_vla.sh           # Package RTL into Xilinx Object (.xo)
    ├── gen_xclbin.sh             # Link .xo into bitstream (.xclbin)
    ├── gen_xo_vla.tcl            # Vivado TCL packaging script
    ├── link_xo_vla_accel.py      # Vitis v++ linker with HBM connectivity
    └── opts.cfg                  # Build configuration options
```

## Architecture

The accelerator processes CNN layers by iterating over spatial strips, output channel tiles (OC_PAR=16), and input channel tiles (IC_PAR=16). Key features:

- **Tile-level sparsity**: The tile manager checks a bitmask before each output tile. Masked tiles are skipped entirely (no DMA, no compute) at 1-cycle cost.
- **Double-buffered pipeline**: Activations in URAM and weights in BRAM use ping-pong banks for DMA/compute overlap.
- **INT8 datapath**: 28-bit internal accumulators with per-layer quantization shift and optional ReLU.
- **Performance counters**: Cycle-accurate decomposition into compute, memory, overlap, and stall cycles.

## NISA Instruction Format

Each instruction is 64 bytes (512 bits):

| Field | Bits | Description |
|-------|------|-------------|
| Memory offsets | 256 | input, output, weight, bias (64b each) |
| Spatial dims | 32 | width, height |
| Feature dims | 32 | in_channels, out_channels |
| Opcode | 8 | CONV, GEMM, MEMCPY, GAP, HALT |
| Quant shift | 8 | Post-accumulation right-shift |
| Bank select | 8 | Ping-pong buffer selection |
| Stride | 8 | Convolution stride |
| Aux flags | 8 | ReLU, flatten, sparse, bias_en |
| Tile masks | 128 | Two 64-bit OC tile bitmasks |

## Build Instructions

### Prerequisites
```bash
export XILINX_XRT=/opt/xilinx/xrt
export XILINX_VITIS=/path/to/Vitis/2022.2
export XILINX_VIVADO=/path/to/Vivado/2022.2
export XILINX_HLS=/path/to/Vitis_HLS/2022.2
```

### Package RTL as .xo
```bash
cd build_scripts
./build_xo_vla.sh
```

### Generate Bitstream
```bash
./gen_xclbin.sh
```

### Build Host
```bash
cd ../host
source /opt/xilinx/xrt/setup.sh
python make_host.py
```

### Run Inference
```bash
./app_main.exe <path_to_xclbin> model_spec.txt
```

## HBM Port Mapping

| AXI Port | HBM Banks | Usage |
|----------|-----------|-------|
| m_axi_gmem | HBM[0:2] | Feature map reads + instruction fetch |
| m_axi_fo_gmem | HBM[1:2] | Output feature map writes |
| m_axi_wgmem | HBM[3] | Weight and bias reads |

## Parallelism Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| OC_PAR | 16 | Output channels per tile |
| IC_PAR | 16 | Input channels per PE |
| PP_PAR | 8 | Pixels processed in parallel |
| DATA_WIDTH | 8 | INT8 activations and weights |
| ACC_WIDTH | 28 | Internal accumulator width |
