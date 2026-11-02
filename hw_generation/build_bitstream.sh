#!/usr/bin/env bash
# =============================================================================
# Sparse by Command (MICRO 2026) -- Artifact: rebuild the bitstream (Table 3)
# Runs the full RTL -> .xo (Vivado package) -> .xclbin (Vitis v++ link) flow.
# This is OPTIONAL: Table 3 can be read from the prebuilt reports shipped with
# the artifact via scripts/reproduce_table3_prebuilt.sh. Rebuilding takes a few
# hours and needs a Vivado/Vitis 2022.1 installation (no FPGA required).
#
# Run inside a long-lived session (tmux/nohup): the build runs to completion,
# then prints where the utilization/timing reports land.
# =============================================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../hw_build" || { echo "ERROR: hw_build/ not found"; exit 1; }

for v in XILINX_VIVADO XILINX_VITIS; do
  [ -n "${!v:-}" ] || { echo "ERROR: \$$v is not set (source the Vivado/Vitis settings)."; exit 1; }
done

echo ">> [1/2] Packaging RTL -> vla_accel.xo (Vivado package_xo) ..."
./build_xo_vla.sh || { echo "ERROR: .xo packaging failed"; exit 1; }

echo ">> [2/2] Linking .xo -> vla_accel.xclbin (Vitis v++, target 300 MHz) ..."
echo "   This is the long step (a few hours: synthesis + place-and-route)."
./gen_xclbin.sh || { echo "ERROR: .xclbin link failed"; exit 1; }

echo
echo "============================================================"
echo "BUILD FINISHED."
echo "Resource utilization (Table 3) report:"
echo "  hw_build/<link_dir>/_x_temp.hw.*/link_report/link/imp/impl_1_hw_bb_locked_utilization_placed.rpt"
echo "Timing/frequency report:"
echo "  hw_build/<link_dir>/_x_temp.hw.*/link_report/link/imp/impl_1_hw_bb_locked_timing_summary_routed.rpt"
echo
echo "Print the Table 3 numbers from that report with:"
echo "  RPT=<path-to-utilization_placed.rpt> bash scripts/reproduce_table3_prebuilt.sh"
echo "============================================================"
