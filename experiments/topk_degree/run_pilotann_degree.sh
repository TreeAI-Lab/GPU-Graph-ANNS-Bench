#!/bin/bash
# Figure 12, PilotANN curves: PilotANN on sift1M with NSG graphs of degree 64 and 128
# (efConstruction 80, d_principle 192, sample ratio 0.25), then an ef sweep of the first 10,000
# queries (per ef one untimed warm-up run and three timed runs, means reported, k = 10).
# Degree 128 needs the larger traversal buffer of third_party/patches/pilotann.patch
# (BUFFER_SIZE 512 >= beam width 4 x degree 128).
# The builder and the sweep are those of the Figure 7 PilotANN curves (src/search/pilotann).
#
# Usage: bash experiments/topk_degree/run_pilotann_degree.sh
# Outputs:
#   $INDEX_DIR/sift1M/pilotann/nsg<D>_efc80_dp192_r0.25/index.pt
#   $OUT_DIR/topk_degree/pilotann/degree<D>/{build.log,run.log,params.txt,load_timing.tsv,sweep.tsv,it_batches.tsv}
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

PILOT=$BENCH/src/search/pilotann
DEGREES=${DEGREES:-64 128}
EFC=80
DP=192
RATIO=0.25
GTYPE=nsg
K=10
BATCH=10000
DATA=$DATA_DIR/sift1M
PY=$ENV_PILOT/bin/python
export LD_LIBRARY_PATH=$ENV_PILOT/lib:${LD_LIBRARY_PATH:-}

for D in $DEGREES; do
  OUT=$OUT_DIR/topk_degree/pilotann/degree$D
  CKPT=$INDEX_DIR/sift1M/pilotann/${GTYPE}${D}_efc${EFC}_dp${DP}_r${RATIO}/index.pt
  mkdir -p "$OUT" "$(dirname "$CKPT")"
  if [[ ! -f "$CKPT" ]]; then
    echo "=== build PilotANN ${GTYPE}${D} -> $CKPT ==="
    "$PY" -u "$PILOT/pilotann_build.py" --base "$DATA/base.fbin" --ckpt "$CKPT" \
      --nn "$D" --efc "$EFC" --dp "$DP" --ratio "$RATIO" --gtype "$GTYPE" 2>&1 | tee "$OUT/build.log"
  fi
  echo "=== sweep degree $D -> $OUT ==="
  "$PY" -u "$PILOT/pilotann_sweep.py" --ckpt "$CKPT" --query "$DATA/query.fbin" --gt "$DATA/gt.ibin" \
    --out "$OUT" --dataset sift1M --batch "$BATCH" --k "$K" 2>&1 | tee "$OUT/run.log"
done
