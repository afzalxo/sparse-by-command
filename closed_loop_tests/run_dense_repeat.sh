#!/bin/bash
source /opt/xilinx/xrt/setup.sh 2>/dev/null
cd ~/vla-circuits-accel/host
cp model_spec_dense.txt model_spec_temp.txt
for i in 1 2 3 4 5; do
  echo "--- Run $i ---"
  ./app_single.exe vla_accel.xclbin 2>&1 | grep -E "Total Execution|Compute Engine|Stall Cycles"
done
