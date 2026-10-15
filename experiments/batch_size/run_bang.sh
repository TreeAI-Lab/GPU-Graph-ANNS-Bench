#!/bin/bash
# Batch-size sweep of BANG on SIFT1M (Figure 8).
# Usage: bash experiments/batch_size/run_bang.sh [batch size ...]   (default: 1 10 100 1000 10000)
# For every batch size bang_search searches the first <batch> queries for L = 10, 22, ..., 502 (step BANG_L_STEP),
# each L once as an untimed warm-up and then three times timed. QPS = batch / mean wall time of the three timed
# bang_query calls, as printed by BANG ("Wall Clock Time", whole milliseconds).
# Output: $OUT_DIR/batch_size/bang/<batch>/ (raw.log, sweep.tsv), $OUT_DIR/batch_size/bang/run.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)
QR=$BENCH/experiments/qps_recall

BATCHES=${*:-"1 10 100 1000 10000"}
BANG_DIR=${BANG_DIR:-$W/build/bang/r128}
PREFIX=$INDEX_DIR/sift1M/bang/r64_l100/index
OUT=$OUT_DIR/batch_size/bang
PY=${PY:-$ENV_BUILD/bin/python}
export BANG_L_STEP=${BANG_L_STEP:-12}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-14}
export LD_LIBRARY_PATH=$BANG_DIR:$ENV_BUILD/lib:${LD_LIBRARY_PATH:-}

[[ -s ${PREFIX}_disk.bin ]] || { echo "missing BANG index ${PREFIX}_disk.bin (run build_indexes.sh)" >&2; exit 2; }
mkdir -p "$OUT"
gt=$OUT_DIR/batch_size/gt.k10.ibin
[[ -f $gt ]] || "$PY" "$QR/gt_top10.py" "$DATA_DIR/sift1M/gt.ibin" "$gt"
: >"$OUT/run.log"
for bs in $BATCHES; do
  mkdir -p "$OUT/$bs"
  echo "START bang bs=$bs $(date '+%F %T')" | tee -a "$OUT/run.log"
  set +e
  "$BANG_DIR/bang_search" "$PREFIX" "$DATA_DIR/sift1M/query.fbin" "$gt" "$bs" 10 float l2 >"$OUT/$bs/raw.log" 2>&1
  rc=$?
  set -e
  echo "DONE bang bs=$bs rc=$rc $(date '+%F %T') lines=$(wc -l <"$OUT/$bs/raw.log")" | tee -a "$OUT/run.log"
  "$PY" "$HERE/parse_bang_batch.py" --raw "$OUT/$bs/raw.log" --out "$OUT/$bs/sweep.tsv" --nq "$bs"
done
echo BANG_ALL_DONE
