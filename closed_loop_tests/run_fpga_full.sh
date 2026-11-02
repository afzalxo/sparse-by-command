#!/bin/bash
source /opt/xilinx/xrt/setup.sh 2>/dev/null
cd ~/vla-circuits-accel/host

EXE=./app_single.exe
XCLBIN=vla_accel.xclbin

SPECS=(
  "dense:model_spec_dense.txt"
  "follow_lane:model_spec_task0_follow_lane.txt"
  "turn_left:model_spec_task1_turn_left.txt"
  "turn_right:model_spec_task2_turn_right.txt"
  "chlane_left:model_spec_task3_chlane_left.txt"
  "chlane_right:model_spec_task4_chlane_right.txt"
  "brake:model_spec_task5_brake.txt"
)

for entry in "${SPECS[@]}"; do
  name="${entry%%:*}"
  spec="${entry##*:}"
  echo "=== $name ==="
  cp "$spec" model_spec_temp.txt
  $EXE $XCLBIN 2>&1 | grep -A20 "N-ISA Hardware Performance Counters"
  echo ""
done
