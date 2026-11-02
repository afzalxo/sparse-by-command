#!/bin/bash
# Open-loop test: runs the CNN driving accelerator on a sample 256x256 image
# for all 6 driving commands and reports steer/throttle.
set +e
cd "$(dirname "$0")"
XCLBIN=${1:-vla_accel.xclbin}
BIN=host/app_open_loop.exe
[ -x "$BIN" ] || { echo "Build first: ./build_host.sh"; exit 1; }
CMDS=(follow_lane turn_left turn_right chlane_left chlane_right brake)

echo "############################################################"
echo "# (A) BIT-EXACT correctness check vs official INT8 goldens  #"
echo "#     (cmd0 masks + branch b{c} -> golden_b{c}_head.bin)     #"
echo "############################################################"
pass=0
for c in 0 1 2 3 4 5; do
  out=$("$BIN" "$XCLBIN" "$c" specs/verify_b${c}.txt 2>&1)
  head=$(echo "$out" | grep -oE 'raw head \(int8\): \[.*\]')
  chk=$(echo "$out" | grep -oE 'golden check.*')
  echo "  branch b${c} (${CMDS[$c]}): $head | $chk"
  echo "$chk" | grep -q PASSED && pass=$((pass+1))
done
echo "  --> $pass/6 branches bit-exact to golden"
echo ""
echo "############################################################"
echo "# (B) SPARSE-BY-COMMAND demo (each command's own masks)     #"
echo "############################################################"
for c in 0 1 2 3 4 5; do
  out=$("$BIN" "$XCLBIN" "$c" specs/spec_cmd${c}.txt 2>&1)
  head=$(echo "$out" | grep -oE 'raw head \(int8\): \[.*\]')
  st=$(echo "$out" | grep -oE 'steer +: .*')
  th=$(echo "$out" | grep -oE 'throttle +: .*')
  printf "  cmd%d %-13s %s  %s  %s\n" "$c" "${CMDS[$c]}" "$head" "$st" "$th"
done
echo ""
echo "Reference values: see expected_results.txt"
