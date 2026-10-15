#!/bin/bash
# Batch-size sweep of PilotANN on SIFT1M (Figure 8), see src/search/pilotann/pilotann_batch.py for the timing.
# Usage: bash experiments/batch_size/run_pilotann.sh [batch size ...]   (default: 1 10 100 1000 10000)
# Output: $OUT_DIR/batch_size/pilotann/<batch>/sweep.tsv, $OUT_DIR/batch_size/pilotann_run.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
QR=$BENCH/experiments/qps_recall
source "$QR/datasets.sh"

BATCHES=${*:-"1 10 100 1000 10000"}
PY=${PY:-$ENV_BUILD/bin/python}
read -r gtype nn efc dp ratio <<<"$(cfg_pilotann sift1M)"
CKPT=$INDEX_DIR/sift1M/pilotann/$(pilotann_tag "$gtype" "$nn" "$efc" "$dp" "$ratio")/index.pt
OUT=$OUT_DIR/batch_size
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-14}
export LD_LIBRARY_PATH=$ENV_PILOT/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$OUT/pilotann"
gt=$OUT/gt.k10.ibin
[[ -f $gt ]] || "$PY" "$QR/gt_top10.py" "$DATA_DIR/sift1M/gt.ibin" "$gt"
set +e
# shellcheck disable=SC2086
"$ENV_PILOT/bin/python" -u "$BENCH/src/search/pilotann/pilotann_batch.py" --ckpt "$CKPT" \
  --query "$DATA_DIR/sift1M/query.fbin" --gt "$gt" --it-file "$QR/it_schedule.txt" \
  --out "$OUT/pilotann" --batches $BATCHES >"$OUT/pilotann_run.log" 2>&1
rc=$?
set -e
echo "pilotann exit $rc $(date '+%F %T')"
exit $rc
