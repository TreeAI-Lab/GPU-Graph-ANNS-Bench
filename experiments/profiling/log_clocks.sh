#!/bin/bash
# Run a command while nvidia-smi records the SM clock, memory clock, power draw and active throttle reasons every
# CLK_LOG_MS milliseconds. The clocks are not locked; the log shows which clocks the measurements ran at.
# usage: bash log_clocks.sh <log dir> <command...>
# output: <log dir>/clocks.csv (timestamp, SM MHz, memory MHz, power W, throttle reasons) and clocks_summary.txt
#   (how often each SM clock was seen, and the maximum power)
set -euo pipefail
LOG=${1:?usage: log_clocks.sh <log_dir> <cmd...>}
shift
mkdir -p "$LOG"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active \
  --format=csv,noheader,nounits -lms "${CLK_LOG_MS:-1000}" > "$LOG/clocks.csv" &
SMI=$!
rc=0
"$@" || rc=$?
kill "$SMI" 2>/dev/null || true
wait "$SMI" 2>/dev/null || true
awk -F', ' '{n[$2]++; if ($4 > p) p = $4} END {for (f in n) printf "sm_MHz %s x%d\n", f, n[f]; printf "max_power_W %.0f\n", p}' \
  "$LOG/clocks.csv" | sort | tee "$LOG/clocks_summary.txt"
exit $rc
