#!/usr/bin/env bash
# =============================================================================
# Sparse by Command (MICRO 2026) -- Artifact reproduction: Table 2
# FPGA on-device latency, cycle decomposition, and active-tile counts on the
# AMD/Xilinx Alveo U50.
#
# For the dense (task-agnostic) configuration and each of the six driving
# commands, this runs one inference of the 8-layer CNN backbone + fused heads
# on the accelerator, reads the on-chip N-ISA performance counters, and reports:
#   - Active tiles   (of the 182 prunable conv tiles), from the instruction masks
#   - Total cycles   (N-ISA "Total Execution Cycles" counter)
#   - Compute cycles (N-ISA "Compute Engine Cycles" counter)
#   - Latency (ms)   = Total cycles / 287 MHz (design Fmax)
#   - Speedup        = dense_total / command_total
#   - Correctness    (INT8 output verified bit-exact against golden reference)
#
# -----------------------------------------------------------------------------
# REPRODUCIBILITY NOTE
# The COMPUTE cycles, ACTIVE-TILE counts, and SPEEDUP ratios are deterministic
# and reproduce exactly. The absolute TOTAL cycles (hence latency) include an
# HBM memory-wait component whose value depends on the HBM PHY timing chosen at
# each card reset (approx +/-3%; two clusters observed: ~2.547M "fast" and
# ~2.62M "slow"). The paper's Table 2 corresponds to the "slow" calibration
# (dense ~2.62M cycles -> 9.12 ms). By default this script resets the card until
# the HBM lands in that state, then measures all seven configurations in one
# consistent state. Set DO_RESET=0 to measure the current state as-is.
# =============================================================================
set -u

# ---- configuration (override via environment) ----
ART_ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # artifact root
HOST_DIR=${HOST_DIR:-"$ART_ROOT/host"}               # host sources + model_spec_*.txt
XCLBIN=${XCLBIN:-vla_accel.xclbin}                    # 287 MHz PP_PAR=8 bitstream
EXE=${EXE:-./app_single.exe}                          # host built from main_carla.cpp
BDF=${BDF:-0000:01:00.1}                              # `xbutil examine` -> user BDF
FREQ_KHZ=${FREQ_KHZ:-287000}                          # design Fmax (287 MHz)
TGT_LO=${TGT_LO:-2600000}                             # accept dense total cycles in [LO,HI]
TGT_HI=${TGT_HI:-2700000}                             #   (paper "slow" HBM calibration)
MAX_RESETS=${MAX_RESETS:-40}
DO_RESET=${DO_RESET:-1}                               # 0 = skip reset loop, measure current state

source /opt/xilinx/xrt/setup.sh 2>/dev/null
cd "$HOST_DIR" || { echo "ERROR: HOST_DIR not found: $HOST_DIR"; exit 1; }

# ---- Stage the prebuilt bitstream + INT8 data into HOST_DIR if not present ----
for asset in vla_accel.xclbin fpga_data_carla; do
  if [ ! -e "$asset" ] && [ -e "$ART_ROOT/prebuilt/$asset" ]; then
    echo ">> Staging $asset from prebuilt/"; cp -r "$ART_ROOT/prebuilt/$asset" .
  fi
done
[ -e "$XCLBIN" ] || { echo "ERROR: $XCLBIN not in $HOST_DIR. Place the prebuilt bitstream here (see prebuilt/MANIFEST.md)."; exit 1; }
[ -d fpga_data_carla ] || { echo "ERROR: fpga_data_carla/ not in $HOST_DIR. Place the INT8 weights/golden here (see prebuilt/MANIFEST.md)."; exit 1; }

# ---- Step 0: build the single-inference host from source (main_carla.cpp) ----
if [ "${BUILD_HOST:-1}" = "1" ] || [ ! -x "$EXE" ]; then
  echo ">> Building host: python3 make_host.py  (main_carla.cpp -> app_single.exe)"
  python3 make_host.py || { echo "ERROR: host build failed"; exit 1; }
fi

specs=(
  "Dense (all tiles):model_spec_dense.txt"
  "Follow Lane:model_spec_task0_follow_lane.txt"
  "Turn Left:model_spec_task1_turn_left.txt"
  "Turn Right:model_spec_task2_turn_right.txt"
  "Ch. Lane Left:model_spec_task3_chlane_left.txt"
  "Ch. Lane Right:model_spec_task4_chlane_right.txt"
  "Brake:model_spec_task5_brake.txt"
)

field() { grep "$2" <<<"$1" | grep -oE '[0-9]+' | head -1; }

# active prunable tiles = sum of popcount(conv mask) over the 8 conv layers, minus
# the 8 always-on tile-0's. Conv mask is field 11 (index 10) of each conv_* line.
active_tiles() {
  python3 -c "
import sys
tot=0
for l in open('$1').read().splitlines():
    f=l.split()
    if f and f[0].startswith('conv_'):
        tot+=bin(int(f[10])).count('1')
print(tot-8)
"
}

# ---- Step 1: reset until HBM lands in the paper (slow) calibration state ----
if [ "$DO_RESET" = "1" ]; then
  echo ">> Resetting card until dense total cycles in [$TGT_LO, $TGT_HI] (paper HBM state)"
  for a in $(seq 1 "$MAX_RESETS"); do
    xbutil reset --device "$BDF" --force >/dev/null 2>&1; sleep 4
    cp model_spec_dense.txt model_spec_temp.txt
    d=$(field "$($EXE $XCLBIN 2>&1)" "Total Execution")
    echo "   [reset $a] dense total = ${d:-?}"
    if [ "${d:-0}" -ge "$TGT_LO" ] && [ "${d:-0}" -le "$TGT_HI" ]; then
      echo "   -> target HBM state reached after $a reset(s)."; break
    fi
    [ "$a" -eq "$MAX_RESETS" ] && echo "   -> WARNING: target not reached in $MAX_RESETS resets; measuring current state."
  done
fi

# ---- Step 2: measure all seven configurations ----
printf "\n%-17s %6s %12s %12s %9s %8s %7s\n" "Command" "Tiles" "TotalCyc" "ComputeCyc" "Lat(ms)" "Speedup" "Verify"
printf '%s\n' "---------------------------------------------------------------------------------"
dense_cyc=0
for entry in "${specs[@]}"; do
  name="${entry%%:*}"; spec="${entry##*:}"
  cp "$spec" model_spec_temp.txt
  out=$($EXE "$XCLBIN" 2>&1)
  tot=$(field "$out" "Total Execution")
  cmp=$(field "$out" "Compute Engine")
  til=$(active_tiles "$spec")
  ver=$(grep -oE "TEST PASSED|TEST FAILED" <<<"$out" | head -1)
  [ "$name" = "Dense (all tiles)" ] && dense_cyc=${tot:-0}
  lat=$(awk "BEGIN{printf \"%.2f\", ${tot:-0}/$FREQ_KHZ}")
  sp=$(awk "BEGIN{ if (${tot:-0}>0 && $dense_cyc>0) printf \"%.2fx\", $dense_cyc/${tot}; else print \"-\" }")
  printf "%-17s %6s %12s %12s %9s %8s %7s\n" "$name" "$til" "${tot:-?}" "${cmp:-?}" "$lat" "$sp" "${ver:-?}"
done
echo
echo "Compare against Table 2. Compute cycles / active tiles / speedups reproduce exactly;"
echo "absolute latency reproduces within ~1-2% in the paper HBM-calibration state."
