# Sparse by Command — Artifact (MICRO 2026)

**Task-Conditional Compute Skipping for Multi-Task Inference Accelerators.**

An INT8 CNN driving accelerator for AMD/Xilinx **Alveo U50**. A gating network maps
each driving command to per-tile execution masks (**sparse-by-command**); the
accelerator skips masked tiles before DMA/compute and zeroes their outputs. It
processes a 256×256×3 image through 8 conv layers → fusion → per-command FC heads,
emitting steering/throttle.

## Directory layout

```
sparse-by-command_artifacts/
├── hw_generation/     Hardware generation: SystemVerilog RTL (rtl/) + build
│                      scripts (build_scripts/: RTL→.xo via Vivado, .xo→.xclbin
│                      via Vitis) + build/timing reports (reports/).
├── sw/                PyTorch model (models/), dense+sparse checkpoints
│                      (checkpoints/), training (train_carla_*.py) and eval recipes,
│                      carla_utils/.
├── compiler/          Data generation for the FPGA: extract_data_for_fpga.py
│                      (quantize weights/activations to INT8, per-layer shifts,
│                      per-command tile masks, golden references) +
│                      gen_task_specs.py + example_specs/.
├── open_loop_tests/   ★ Ready-to-run FPGA test: prebuilt xclbin, host, a sample
│                      image, weights, per-command specs, and run scripts. See
│                      open_loop_tests/README.md.
├── closed_loop_tests/ CARLA closed-loop driving (continuous host + eval). Provided
│                      for a later step; not required for the on-device claims.
└── README.md
```

## Quick start — open-loop FPGA test

The fastest way to see the accelerator working (needs a U50 + XRT + Vitis HLS):

```bash
cd open_loop_tests
source /opt/xilinx/xrt/setup.sh
export XILINX_HLS=/path/to/Xilinx/Vitis_HLS/2022.2
./build_host.sh
./run_open_loop.sh
```

This runs the sample image through all six driving commands and reports:
- **(A)** bit-exact correctness vs the extracted INT8 goldens (6/6 branches PASS), and
- **(B)** the sparse-by-command steer/throttle for each command.

See `open_loop_tests/README.md` for details and expected output.

## Reproducing the paper tables

- **Active tiles per command** (Table 2, software): `cd sw && python3 extract_active_tiles.py` → 49/57/55/54/53/34.
- **On-device latency** (Table 2, FPGA): `open_loop_tests/reproduce_table2.sh`.
- **Resource / frequency** (Table 3): `open_loop_tests/reproduce_table3_prebuilt.sh`, or rebuild with `hw_generation/build_bitstream.sh`.

Key parallelism (paper): `OC_PAR=IC_PAR=16`, `PP_PAR=8` → 2048 INT8 MACs/cycle
(`open_loop_tests/host/srcs/globals.hpp`).

## Fixes included in this revision

This artifact packages the corrected, bit-exact design:

1. **conv_7 tiling (host)** — clamp `log2_mem_tile_height` to the input height for
   tiny layers (`in_h < mem_tile_h`, i.e. conv_7's 2×2 input). Without it the
   `tiled_dma` IC-tile stride misaligns conv_7's inputs. Applied in all hosts.
2. **masked-tile zeroing (RTL, `S_ZERO_MASKED` in tile_manager.sv + `zero_fill` in
   output_dma.sv)** — masked oc-tile output slots are zeroed (compute still
   skipped) so no stale garbage leaks into the fusion.
3. **build config** — the linker passes `--config opts.cfg` (Explore/phys-opt P&R
   directives); kernel closes at 261.7 MHz.

The shipped `open_loop_tests/vla_accel.xclbin` is built from `hw_generation/rtl/`
and validated **bit-exact** to the INT8 software reference across all conv →
fusion → FC → head stages for all six command branches.
