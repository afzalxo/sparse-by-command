#!/bin/bash
# Build the open-loop host (app_open_loop.exe). Requires XRT + Vitis HLS.
set -e
: "${XILINX_XRT:?set XILINX_XRT (e.g. source /opt/xilinx/xrt/setup.sh)}"
: "${XILINX_HLS:?set XILINX_HLS (e.g. export XILINX_HLS=/tools/Xilinx/Vitis_HLS/2022.2)}"
cd "$(dirname "$0")/host"
FPO=${XILINX_HLS}/lnx64/tools/fpo_v7_1
g++ -o app_open_loop.exe main_open_loop.cpp xcl2.cpp srcs/*.cpp \
  -fmessage-length=0 -I${XILINX_XRT}/include -I${XILINX_HLS}/include -std=c++14 -O3 \
  -Wall -Wno-unknown-pragmas -Wno-unused-label -I . \
  -L${XILINX_XRT}/lib -L${FPO} -Wl,-rpath,${FPO} -Wl,-rpath,${XILINX_XRT}/lib \
  -lOpenCL -lxrt_coreutil -lgmp -lmpfr -lIp_floating_point_v7_1_bitacc_cmodel
echo "Built host/app_open_loop.exe"
