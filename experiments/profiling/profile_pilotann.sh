#!/bin/bash
# Nsight Compute for PilotANN on DEEP1M and GIST, the two Table 9 groups next to the 18 of profile_all.sh
# (run_all.sh does not call this script). For each dataset:
#   1. run_point.sh at the operating point of points.tsv; stops if Recall@10 is not a number of at least 0.95
#   2. find_kernel.sh; the launches per search are the instances of the kernels matching kernel_regex of
#      points.tsv divided by the 4 searches of run_point.sh (one warm-up, three timed); stops if not a whole number
#   3. profile_ncu.sh on all launches of the first timed search, under log_clocks.sh
# and then summarize.py over every group under $OUT_DIR/profiling.
# Needs $ENV_PILOT with PilotANN built for the A10 without the kernel statistics of the patch
# (PILOT_NO_KERNEL_STATS=1 TORCH_CUDA_ARCH_LIST=8.6 bash third_party/build/pilotann.sh); the script stops if the
# installed extension still reads the clock (SR_CLOCKLO in its SASS). It also needs the Figure 7 PilotANN indexes
# (PILOT_CKPT_DEEP1M and PILOT_CKPT_GIST point to other copies, see run_point.sh).
# usage: bash profile_pilotann.sh [deep1M] [gist]     (no argument: both)
# output: $OUT_DIR/profiling/<dataset>_pilotann/{point,nsys,ncu}/, ncu_log_pilotann/<dataset>/clocks.csv,
#   summary.tsv
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
P=$OUT_DIR/profiling
PYTHON=${PYTHON:-python3}
export PREC=fp32 CLK_LOG_MS=${CLK_LOG_MS:-200}
DATASETS=("$@")
[[ ${#DATASETS[@]} -gt 0 ]] || DATASETS=(deep1M gist)
nvidia-smi -rgc > /dev/null || echo "warning: nvidia-smi -rgc failed; check that the GPU clocks are not locked"
so=$(LD_LIBRARY_PATH=$ENV_PILOT/lib:${LD_LIBRARY_PATH:-} "$ENV_PILOT/bin/python" -c "import pilot_ann.extension as x; print(x.__file__)")
CUOBJDUMP=${CUOBJDUMP:-cuobjdump}   # CUDA 11.1 or newer, to disassemble sm_80 and sm_86
command -v "$CUOBJDUMP" > /dev/null || { echo "cuobjdump not found (set CUOBJDUMP) to check $so"; exit 1; }
sass=$("$CUOBJDUMP" -sass "$so" 2>&1) || true
grep -q "Function : .*traverse_cuda_kernel" <<< "$sass" || { echo "$CUOBJDUMP could not disassemble $so (too old?)"; exit 1; }
nclk=$(grep -c SR_CLOCKLO <<< "$sass" || true)
[[ "$nclk" == 0 ]] || { echo "$so reads the clock ($nclk SR_CLOCKLO): rebuild with PILOT_NO_KERNEL_STATS=1"; exit 1; }
echo "PilotANN extension without kernel statistics: $so"
for ds in "${DATASETS[@]}"; do
  [[ "$ds" == deep1M || "$ds" == gist ]] || { echo "unknown dataset $ds (deep1M gist)" >&2; exit 1; }
  ref=$(awk -F'\t' -v d="$ds" '$1==d && $2=="pilotann" {print $4}' "$HERE/points.tsv")
  kregex=$(awk -F'\t' -v d="$ds" '$1==d && $2=="pilotann" {print $6}' "$HERE/points.tsv")

  echo "== $ds pilotann point $(date -Is)"
  pt=$P/${ds}_pilotann/point
  mkdir -p "$pt"
  bash "$HERE/run_point.sh" "$ds" pilotann "$pt" > "$pt/run_point.log" 2>&1 || { echo "FAILED, see $pt/run_point.log"; exit 1; }
  rec=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++)c[$i]=i} NR==2{print $c["recall@10"]}' "$pt/sweep.tsv")
  echo "recall@10 $rec (reference $ref)"
  if ! [[ "$rec" =~ ^[0-9]+(\.[0-9]+)?$ ]] || ! awk -v r="$rec" 'BEGIN { exit !(r >= 0.95) }'; then
    echo "recall '$rec' is below 0.95 or not a number (see $pt/pilotann.log): set another ef in points.tsv"
    exit 1
  fi

  echo "== $ds pilotann kernels $(date -Is)"
  bash "$HERE/find_kernel.sh" "$ds" pilotann
  inst=$("$PYTHON" - "$P/${ds}_pilotann/nsys/kern_sum.csv" "$kregex" <<'EOF'
import csv, re, sys
print(sum(int(r['Instances']) for r in csv.DictReader(open(sys.argv[1])) if re.search(sys.argv[2], r['Name'])))
EOF
)
  [[ "$inst" =~ ^[0-9]+$ && "$inst" -gt 0 && $((inst % 4)) -eq 0 ]] || { echo "$kregex instances '$inst' not a positive multiple of 4"; exit 1; }
  launches=$((inst / 4))
  echo "launches per search: $launches"

  echo "== $ds pilotann ncu $(date -Is)"
  NCU_LAUNCH_SKIP=$launches NCU_LAUNCH_COUNT=$launches \
    bash "$HERE/log_clocks.sh" "$P/ncu_log_pilotann/$ds" bash "$HERE/profile_ncu.sh" "$ds" pilotann
done
"$PYTHON" "$HERE/summarize.py" "$P"
awk -F'\t' 'NR==1 || $2=="pilotann"' "$P/summary.tsv"
echo "PROFILE_PILOTANN_DONE $(date -Is)"
