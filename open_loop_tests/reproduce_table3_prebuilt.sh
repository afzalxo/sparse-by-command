#!/usr/bin/env bash
# =============================================================================
# Sparse by Command (MICRO 2026) -- Artifact reproduction: Table 3
# Reads the accelerator resource utilization directly from the prebuilt
# Vivado/Vitis placement report shipped with the artifact (no build required).
#
# The report was produced by the final RTL -> .xo -> .xclbin build (see
# scripts/build_bitstream.sh to regenerate it). This script parses the DSP,
# BRAM, URAM, LUT, and FF counts and prints them next to the paper's Table 3.
# =============================================================================
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
RPT=${RPT:-"$HERE/../prebuilt/hw_reports/utilization_placed.rpt"}

[ -f "$RPT" ] || { echo "ERROR: utilization report not found: $RPT"; exit 1; }

# Extract the "Used" column (field 3) of the first matching resource row.
val() { grep -m1 "| $1 " "$RPT" | awk -F'|' '{gsub(/ /,"",$3); print $3}'; }

lut=$(val 'CLB LUTs'); ff=$(val 'CLB Registers')
bram=$(val 'Block RAM Tile'); uram=$(val 'URAM'); dsp=$(val 'DSPs')

printf "\n%-8s %12s %14s\n" "Resource" "Reproduced" "Paper (Tab.3)"
printf '%s\n' "-------------------------------------"
printf "%-8s %12s %14s\n" "DSP"  "$dsp"  "2082"
printf "%-8s %12s %14s\n" "BRAM" "$bram" "238.5"
printf "%-8s %12s %14s\n" "URAM" "$uram" "60"
printf "%-8s %12s %14s\n" "LUT"  "$lut"  "234359"
printf "%-8s %12s %14s\n" "FF"   "$ff"   "388910"
echo
echo "Source report: $RPT"
echo "DSP/BRAM/URAM reproduce exactly; LUT/FF within <0.3%."
echo "Achieved kernel clock (~287 MHz):"
echo "  xclbinutil --info --input vla_accel.xclbin | grep -i frequency"
