# sw — PyTorch model, checkpoints, training & eval

The software side of the sparse-by-command driving model: the network definition,
trained checkpoints, and the CARLA training/eval recipes.

```
sw/
├── models/
│   ├── carla_model.py        CarlaVLAHW_Gated: 8-conv backbone + fusion + 6 FC
│   │                         command heads + TileGatingNetwork (per-tile masks).
│   └── carla_model_pruned.py pruned/dense variant
├── checkpoints/
│   ├── carla_vla_furthest_driver_phase3_gated.pth   sparse (gated) — used everywhere
│   └── carla_vla_dense.pth                           dense baseline
├── train_carla_sparse.py     train the gated (sparse-by-command) model
├── train_carla_baseline.py   train the dense baseline
├── eval_fpga.py, eval_multi_run_static.py, record_dashboard_eval_fpga.py
├── extract_active_tiles.py   report active tiles/command (Table 2, software)
└── carla_utils/              dataset / route / eval helpers
```

## Requirements

`python3` with `torch`, `numpy`, `pandas`, `opencv-python`, `pillow`, `tqdm`.
`carla_utils` pulls in the `carla` module for the *closed-loop* helpers only; the
model itself (`models/carla_model.py`) is pure PyTorch and imports with no CARLA.

## Active-tile reproduction (no FPGA, no CARLA)

```bash
python3 extract_active_tiles.py
# -> per-command active conv tiles: 49/57/55/54/53/34  (of 182 prunable tiles)
```

## How the pieces connect

- **This directory** defines and trains the model.
- **`../compiler/`** quantizes a checkpoint to INT8 and exports `fpga_data/`
  (weights, biases, goldens, input image, per-command tile masks) for the FPGA.
- **`../open_loop_tests/`** runs the exported data on the U50 bitstream.
- **`../closed_loop_tests/`** drives CARLA on-device (later).
