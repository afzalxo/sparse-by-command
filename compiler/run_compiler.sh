#!/bin/bash
# Regenerate the FPGA INT8 data + per-command tile-mask specs from the PyTorch
# checkpoint and a CARLA base-dataset sample.
#
# Requirements (run this where the dataset + GPU/CPU torch live, e.g. a GPU node):
#   python3 with: torch, numpy, pandas, opencv-python (cv2), pillow, tqdm
#   the base CARLA dataset (data.csv + images/) -- NOT bundled; set FPGA_DATASET_DIR.
#
# Env overrides:
#   FPGA_DATASET_DIR  (required)  base dataset dir containing data.csv and images/
#   FPGA_MODEL_PATH   (default ../sw/checkpoints/carla_vla_furthest_driver_phase3_gated.pth)
#   FPGA_OUTPUT_DIR   (default ../open_loop_tests/fpga_data)
#   FPGA_SPEC_DIR     (default ../open_loop_tests/specs)
#   SAMPLE_IDX        (default 2332)
set -e
cd "$(dirname "$0")"
export PYTHONPATH="$(cd ../sw && pwd):${PYTHONPATH}"           # model + carla_utils
MODEL=${FPGA_MODEL_PATH:-../sw/checkpoints/carla_vla_furthest_driver_phase3_gated.pth}
: "${FPGA_DATASET_DIR:?Set FPGA_DATASET_DIR to the base CARLA dataset dir (data.csv + images/)}"
OUT=${FPGA_OUTPUT_DIR:-../open_loop_tests/fpga_data}
SPECS=${FPGA_SPEC_DIR:-../open_loop_tests/specs}
SAMPLE=${SAMPLE_IDX:-2332}
DATANAME=$(basename "$OUT")

echo "[1/3] Extracting INT8 weights/biases/goldens + input image (sample $SAMPLE) -> $OUT"
python3 extract_data_for_fpga.py \
    --model-path "$MODEL" --dataset-dir "$FPGA_DATASET_DIR" \
    --output-dir "$OUT" --sample_idx "$SAMPLE"

echo "[2/3] Generating dense + per-command specs -> $SPECS"
python3 gen_task_specs.py --data-dir "$OUT" --spec-dir "$SPECS" --data-name "$DATANAME"

echo "[3/3] Generating bit-exact verify specs (cmd0 masks + branch b{c}) -> $SPECS"
for c in 0 1 2 3 4 5; do
    head -n 10 "$SPECS/spec_cmd0.txt"      >  "$SPECS/verify_b${c}.txt"
    sed -n '11,13p' "$SPECS/spec_cmd${c}.txt" >> "$SPECS/verify_b${c}.txt"
done

echo ""
echo "Done. Regenerated:"
echo "  $OUT/    weights_*.bin bias_*.bin golden_*.bin input_vision.bin extra_features.bin"
echo "           global_shifts.json model_spec.txt task_masks.txt"
echo "  $SPECS/  spec_cmd{0..5}.txt verify_b{0..5}.txt model_spec_dense.txt"
echo "Now run: (cd ../open_loop_tests && ./build_host.sh && ./run_open_loop.sh)"
