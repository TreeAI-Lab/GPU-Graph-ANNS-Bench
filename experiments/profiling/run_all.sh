#!/bin/bash
# All measurements of Tables 9 and 10 and Figure 14, in order, with the GPU clocks not locked:
#   1. Nsight Compute on the 18 groups (profile_all.sh), then summarize.py -> summary.tsv
#   2. Recall@10 and QPS of Table 10: one run per group, 2 s sleep before it; the run searches the operating point
#      once as an untimed warm-up and three times timed (qps_repeat.sh)
#   3. QPS-recall curves of Figure 14: one sweep per curve, 0.5 s sleep before each iteration count, which is
#      searched once as an untimed warm-up and three times timed (qps_curve.sh)
# nvidia-smi logs the clocks and power every 200 ms during each step (log_clocks.sh).
# usage: bash run_all.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
P=$OUT_DIR/profiling
PYTHON=${PYTHON:-python3}
export CLK_LOG_MS=${CLK_LOG_MS:-200}
mkdir -p "$P"
# Undo any clock lock left from earlier work (needs root)
nvidia-smi -rgc > /dev/null || echo "warning: nvidia-smi -rgc failed; check that the GPU clocks are not locked"
echo "== step 1 ncu start $(date -Is)"
bash "$HERE/log_clocks.sh" "$P/ncu_log" bash "$HERE/profile_all.sh"
"$PYTHON" "$HERE/summarize.py" "$P"
echo "== step 2 table 10 qps start $(date -Is)"
OUT=$P/table10_qps PRE_SLEEP_S=2 bash "$HERE/log_clocks.sh" "$P/table10_qps" bash "$HERE/qps_repeat.sh" 1
echo "== step 3 curves start $(date -Is)"
OUT=$P/fp16_curves REPS=1 IT_SLEEP_MS=500 bash "$HERE/log_clocks.sh" "$P/fp16_curves" bash "$HERE/qps_curve.sh"
echo "RUN_ALL_DONE $(date -Is)"
