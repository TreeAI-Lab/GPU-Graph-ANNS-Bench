#!/bin/bash
# QPS-recall sweep of PilotANN over ef_search (Figure 7), see src/search/pilotann/pilotann_sweep.py for the timing.
# Usage: bash experiments/qps_recall/run_pilotann.sh [dataset ...]   (default: PILOTANN_DATASETS of datasets.sh)
# Output: $OUT_DIR/qps_recall/<dataset>/pilotann/ (sweep.tsv, it_batches.tsv, load_timing.tsv, params.txt, run.log)
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/datasets.sh"

OUT=$OUT_DIR/qps_recall
PY=${PY:-$ENV_BUILD/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-14}
export LD_LIBRARY_PATH=$ENV_PILOT/lib:${LD_LIBRARY_PATH:-}

run_one() {
  local ds=$1 data n dim deg nq rest gtype nn efc dp ratio ckpt gt out rc
  read -r data n dim deg nq rest <<<"$(cfg_graph "$ds")"
  read -r gtype nn efc dp ratio <<<"$(cfg_pilotann "$ds")"
  ckpt=$INDEX_DIR/$data/pilotann/$(pilotann_tag "$gtype" "$nn" "$efc" "$dp" "$ratio")/index.pt
  mkdir -p "$OUT/$ds"
  gt=$OUT/$ds/gt.k10.ibin
  [[ -f $gt ]] || "$PY" "$HERE/gt_top10.py" "$DATA_DIR/$data/gt.ibin" "$gt"
  out=$OUT/$ds/pilotann
  mkdir -p "$out"
  echo "======== pilotann $ds $(date '+%F %T') ========"
  set +e
  "$ENV_PILOT/bin/python" -u "$BENCH/src/search/pilotann/pilotann_sweep.py" --dataset "$ds" --ckpt "$ckpt" \
    --query "$DATA_DIR/$data/query.fbin" --gt "$gt" --out "$out" --batch "$nq" \
    > >(tee "$out/run.log") 2>&1
  rc=$?
  set -e
  echo "DONE $ds pilotann rc=$rc $(date '+%F %T')"
  return $rc
}

fail=0
for ds in ${*:-$PILOTANN_DATASETS}; do
  run_one "$ds" || fail=1
done
exit $fail
