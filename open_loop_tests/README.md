# Open-Loop Tests — Sparse-by-Command CNN Driving Accelerator

This directory lets you run the INT8 CNN driving accelerator **on the FPGA** for a
single sample image, across all six driving commands, and verify the outputs
(steering / throttle) against the software reference — without needing CARLA.

The accelerator processes a 256×256×3 RGB image through 8 convolution layers, a
fusion layer, and per-command fully-connected heads. A gating network maps each
driving command to per-tile execution masks (**sparse-by-command**); masked tiles
are skipped in compute and their outputs zeroed.

## Contents

```
open_loop_tests/
├── vla_accel.xclbin        Prebuilt U50 bitstream (kernel 261.7 MHz)
├── host/                   OpenCL/XRT host
│   ├── main_open_loop.cpp  Single-image driver: packs data, builds the NISA
│   │                       instruction stream, runs the kernel, dequantizes head
│   ├── srcs/               globals.hpp, packing.hpp, utils.*, verify_output.*
│   └── xcl2.*, make_host.py
├── fpga_data/              Sample INT8 weights, biases, the input image
│   │                       (input_vision.bin, 256×256×3), extra_features.bin,
│   │                       golden_*.bin references, global_shifts.json
├── specs/
│   ├── spec_cmd{0..5}.txt  Per-command model specs WITH each command's own
│   │                       gating masks (sparse-by-command)
│   └── verify_b{0..5}.txt  cmd0 masks + branch b{c} (for bit-exact golden check)
├── expected_results.txt    Reference INT8 heads + dequantized control values
├── build_host.sh           Compile the host
└── run_open_loop.sh        Build-free test runner (both checks below)
```

## Prerequisites

- AMD/Xilinx **Alveo U50** with XRT installed and the card programmed-capable.
- **XRT** and **Vitis HLS 2022.2** (for the `Ip_floating_point` cmodel the host links).

```bash
source /opt/xilinx/xrt/setup.sh
export XILINX_HLS=/path/to/Xilinx/Vitis_HLS/2022.2
```

## Run

```bash
./build_host.sh          # -> host/app_open_loop.exe
./run_open_loop.sh       # runs both checks below on vla_accel.xclbin
```

`run_open_loop.sh` performs two checks:

**(A) Bit-exact correctness.** For each branch it runs cmd0's convolution masks +
branch `b{c}` and compares the head to the official extracted golden
`golden_b{c}_head.bin`. All six branches are **bit-exact** to the INT8 software
reference:

```
b0 [25,64]  b1 [-9,27]  b2 [15,39]  b3 [1,52]  b4 [13,53]  b5 [0,-65]   (PASSED)
```

**(B) Sparse-by-command demo.** For each command it runs that command's **own**
gating masks (`spec_cmd{c}.txt`) and prints the raw INT8 head plus dequantized
`steer`/`throttle`:

```
cmd0 follow_lane  [25, 64]   steer +0.2124  throttle +0.5022
cmd1 turn_left    [-6, 27]   steer -0.1224  throttle +0.5031
cmd2 turn_right   [14, 39]   steer +0.1943  throttle +0.4993
cmd3 chlane_left  [-7, 52]   steer -0.0735  throttle +0.4983
cmd4 chlane_right [10, 54]   steer +0.1018  throttle +0.5016
cmd5 brake        [0, -65]   steer +0.0000  throttle -0.9998
```

`steer, throttle = tanh(raw_int8 / OUTPUT_SCALES[cmd])` (scales in
`host/srcs/globals.hpp` and `expected_results.txt`).

## Notes

- **Why two checks?** The bundled goldens (`golden_b{c}_head.bin`) were extracted
  from a single command's masks (cmd0), so the bit-exact check (A) holds each
  branch to those official goldens. Check (B) exercises the *sparse-by-command*
  feature, where each command uses its own gating masks — its outputs were
  validated bit-exact against the INT8 software reference (see the artifact-level
  README / `compiler/`).
- Every INT8 stage (conv → fusion → FC → head) is bit-exact between the FPGA and
  the software INT8 reference.
- To run a single command manually:
  `cd open_loop_tests && ./host/app_open_loop.exe vla_accel.xclbin 1 specs/spec_cmd1.txt`
  (arg 2 is the command id 0–5; the spec selects masks + branch).

## Rebuilding the bitstream

The `.xclbin` here is prebuilt. To regenerate it from RTL, see
`../hw_generation/` (RTL + `build_scripts/`). To regenerate `fpga_data/`
(quantized weights, masks, goldens) from the PyTorch checkpoint, see
`../compiler/` and `../sw/`.
