#!/bin/bash
# L sweep of BANG on one dataset, to choose its operating point in points.tsv: the first L (step 8, from 10) with
# Recall@10 >= 0.95. bang_search without an L argument sweeps L itself and runs four searches per L: one untimed
# warm-up and three timed.
# usage: bash bang_sweep.sh <deep1M|gist>
# output: $OUT_DIR/profiling/bang_sweep/<ds>/raw.log and sweep.tsv (L, mean wall ms, QPS = nq / mean wall time,
#   mean Recall@10 in percent, wall ms standard deviation, number of timed runs, as BANG prints them for each L)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
DS=${1:?usage: bang_sweep.sh <deep1M|gist>}
PY=${PY:-$ENV_BUILD/bin/python}
export BANG_L_STEP=${BANG_L_STEP:-8}
case "$DS" in
  deep1M) R=32; NQ=10000 ;;
  gist) R=128; NQ=1000 ;;
  *) echo "unknown dataset $DS" >&2; exit 1 ;;
esac
OUT=$OUT_DIR/profiling/bang_sweep/$DS
GT=$OUT_DIR/profiling/gt/${DS}_k10.ibin
mkdir -p "$OUT" "$(dirname "$GT")"
[[ -f "$GT" ]] || "$PY" "$HERE/make_gt_k10.py" "$DATA_DIR/$DS/gt.ibin" "$GT"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-14} "$W/build/bang/r$R/bang_search" \
  "$INDEX_DIR/$DS/bang/r${R}_l100_b4/bang_format/$DS" "$DATA_DIR/$DS/query.fbin" "$GT" "$NQ" 10 float l2 \
  > "$OUT/raw.log" 2>&1
awk 'NF == 6 && $1 ~ /^[0-9]+$/ && $4 ~ /^[0-9.]+$/ && $6 ~ /^[0-9]+$/' "$OUT/raw.log" > "$OUT/sweep.tsv"
awk '$4 >= 95 { print "first L with Recall@10 >= 0.95: " $1 " (recall " $4 "%, " $3 " QPS)"; exit }' "$OUT/sweep.tsv"
