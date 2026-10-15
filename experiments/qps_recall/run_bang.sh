#!/bin/bash
# QPS-recall sweep of BANG over its worklist length L (Figure 7).
# Usage: bash experiments/qps_recall/run_bang.sh [dataset ...]   (default: BANG_DATASETS of datasets.sh)
# bang_search runs all queries of the dataset as one batch for L = 10, 10 + BANG_L_STEP, ... up to 512.
# For each L it searches four times in a row: one untimed warm-up run, then three timed runs. The wall time of a
# run is one bang_query call (host clock, whole ms); sweep.tsv has the mean over the three timed runs and
# QPS = nq / mean wall time.
# Output: $OUT_DIR/qps_recall/<dataset>/bang/ (raw.log, sweep.tsv, params.txt)
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/datasets.sh"

# BANG patched for the indexes of this study (degree R up to MAX_R = 128), built by third_party/build/bang.sh 128
BANG_DIR=${BANG_DIR:-$W/build/bang/r128}
OUT=$OUT_DIR/qps_recall
PY=${PY:-$ENV_BUILD/bin/python}
export BANG_L_STEP=${BANG_L_STEP:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-14}
export LD_LIBRARY_PATH=$BANG_DIR:$ENV_BUILD/lib:${LD_LIBRARY_PATH:-}

run_one() {
  local ds=$1 data n dim deg nq rest R L M B QD prefix gt out rc nrows
  read -r data n dim deg nq rest <<<"$(cfg_graph "$ds")"
  read -r R L M B QD <<<"$(cfg_bang "$ds")"
  prefix=$INDEX_DIR/$data/bang/$(bang_tag "$R" "$L")/index
  [[ -s ${prefix}_disk.bin ]] || { echo "missing BANG index ${prefix}_disk.bin" >&2; return 2; }
  mkdir -p "$OUT/$ds"
  gt=$OUT/$ds/gt.k10.ibin
  [[ -f $gt ]] || "$PY" "$HERE/gt_top10.py" "$DATA_DIR/$data/gt.ibin" "$gt"
  out=$OUT/$ds/bang
  mkdir -p "$out"
  echo "======== bang $ds step=$BANG_L_STEP runs_per_L=4 (1 warm-up + 3 timed) $(date '+%F %T') ========"
  set +e
  # 7 arguments: automatic L sweep
  "$BANG_DIR/bang_search" "$prefix" "$DATA_DIR/$data/query.fbin" "$gt" "$nq" 10 float l2 >"$out/raw.log" 2>&1
  rc=$?
  set -e
  echo "bang_exit=$rc"
  "$PY" "$HERE/parse_bang_raw.py" --raw "$out/raw.log" --out "$out" --nq "$nq"
  nrows=$(awk 'NR>1{c++} END{print c+0}' "$out/sweep.tsv")
  {
    echo "algo	bang"
    echo "dataset	$ds"
    echo "index_prefix	$prefix"
    echo "batch_size	$nq"
    echo "k	10"
    echo "mode	auto_L_step${BANG_L_STEP}_to_MAX_L512"
    echo "runs_per_L	4"
    echo "warmup_per_L	1"
    echo "timed_per_L	3"
    echo "omp_num_threads	$OMP_NUM_THREADS"
    echo "n_sweep_rows	$nrows"
    echo "exit_code	$rc"
    echo "finished_at	$(date '+%F %T')"
  } >"$out/params.txt"
  echo "======== bang $ds done rows=$nrows $(date '+%F %T') ========"
  [[ $nrows -ge 5 ]] || { echo "ERROR: too few BANG sweep rows ($nrows)" >&2; return 3; }
}

fail=0
for ds in ${*:-$BANG_DATASETS}; do
  run_one "$ds" || fail=1
done
exit $fail
