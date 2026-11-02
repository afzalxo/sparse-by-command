# closed_loop_tests — CARLA on-device driving (work in progress)

Closed-loop evaluation: the FPGA runs inference each frame inside a live CARLA
simulation, feeding steering/throttle back to the vehicle. Unlike `open_loop_tests`
(single fixed image), this requires a running CARLA server.

```
closed_loop_tests/
├── host/
│   ├── main_carla_continuous.cpp   continuous-inference host (streams frames,
│   │                               keeps weights resident across frames)
│   ├── main_carla.cpp              single-frame driver
│   ├── srcs/, xcl2.*, make_host.py
├── eval_fpga.py                    drives CARLA, calls the FPGA host per frame
├── record_dashboard_eval_fpga.py   eval + dashboard/video recording
└── run_fpga_full.sh, run_fpga_cycles.sh, run_dense_repeat.sh
```

## Status

The host and eval scripts are included (with the conv_7 `mem_tile_height` fix
applied), **but this closed-loop flow is not yet wired end-to-end in this artifact.**
It needs:

- a running **CARLA server** + the `carla` Python package (`carla_utils` imports it),
- the closed-loop host built like the open-loop one
  (`export XILINX_HLS=...; source XRT; cd host && ... make_host.py`),
- `eval_fpga.py` pointed at the CARLA server and the built host + `vla_accel.xclbin`.

CARLA rendering/physics are non-deterministic across GPUs/hosts, so absolute
cross-track error is not portably reproducible; the deterministic on-device claims
are covered by `../open_loop_tests/`. This directory will be completed in a later
step.

## For a working, deterministic FPGA demo now

Use `../open_loop_tests/` — it runs the accelerator on a bundled sample image for
all six commands and checks the outputs bit-exact, with no CARLA dependency.
