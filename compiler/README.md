# compiler — INT8 data generation for the FPGA

Turns a trained PyTorch checkpoint (`../sw/checkpoints/`) + one CARLA dataset
sample into everything the FPGA host needs: INT8 weights/biases, quantization
shifts, golden references, the packed input image, the per-command tile masks,
and the model-spec files.

```
compiler/
├── extract_data_for_fpga.py   run the model on one sample; quantize weights/
│                              activations to INT8; compute per-layer shifts and
│                              golden outputs; run the gating net for all 6
│                              commands -> per-command tile masks. Writes
│                              fpga_data/ + model_spec.txt + task_masks.txt.
├── gen_task_specs.py          model_spec.txt + task_masks.txt -> spec_cmd{0..5}.txt
│                              (each keeps its own branch) + model_spec_dense.txt.
├── run_compiler.sh            orchestrates both + builds the verify_b{c}.txt specs.
└── example_specs/             sample generated specs for reference.
```

## Requirements

`python3` with `torch, numpy, pandas, opencv-python (cv2), pillow, tqdm`, run
where the CARLA **base** dataset lives (e.g. a GPU node). The `carla` module is
**not** required — `extract_data_for_fpga.py` falls back to a pure-Python
`command_to_branch_id` if `carla_utils` can't import.

## Regenerate `fpga_data` + specs

```bash
export FPGA_DATASET_DIR=/path/to/carla_base_dataset      # must contain data.csv + images/
# optional overrides:
#   FPGA_MODEL_PATH  (default ../sw/checkpoints/carla_vla_furthest_driver_phase3_gated.pth)
#   FPGA_OUTPUT_DIR  (default ../open_loop_tests/fpga_data)
#   FPGA_SPEC_DIR    (default ../open_loop_tests/specs)
#   SAMPLE_IDX       (default 2332)
bash run_compiler.sh
```

This writes:
- `../open_loop_tests/fpga_data/` — `weights_*.bin`, `bias_*.bin`, `golden_*.bin`,
  `input_vision.bin` (256×256×3), `extra_features.bin`, `global_shifts.json`,
  `model_spec.txt`, `task_masks.txt`
- `../open_loop_tests/specs/` — `spec_cmd{0..5}.txt`, `verify_b{0..5}.txt`,
  `model_spec_dense.txt`

Then run the open-loop test:
`(cd ../open_loop_tests && ./build_host.sh && ./run_open_loop.sh)`.

## Notes

- Quantization matches the RTL bit-exactly: weights `round(w · 127/max|w|)`,
  per-layer right-shift `(acc + 2^(s-1)) >> s` (floor), INT8 clamp `[-128,127]`.
- The shipped `spec_cmd{c}.txt` were produced by exactly this `gen_task_specs.py`
  (verified byte-identical), so re-running the compiler on the same sample
  reproduces the bundled data.
