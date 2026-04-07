#!/bin/sh

target=hw
fpga=u50

rm -rf link_${fpga}_${target}_2025
# Remove previous softFPGA files
rm -rf ../removeme/tmp0
rm ../removeme/xo/*

mkdir -p ./link_${fpga}_${target}_2025/

bash build_xo_vla.sh
cp ../removeme/xo/vla_accel.xo ./link_${fpga}_${target}_2025/

rm -rf ../removeme/tmp0
rm ../removeme/xo/*

python3 link_xo_vla_accel.py --fpga ${fpga} --target $target
# Remove junk files
rm -rf vivado_*
rm -rf v++_vla_accel_*.log
rm -rf xvlog*
rm -rf ./vivado.jou
rm vivado.log
rm xcd.log
