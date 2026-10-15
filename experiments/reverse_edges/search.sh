#!/bin/bash
# Figure 16, step 3: search both graphs with the same settings and iteration budgets.
# Each graph becomes a CAGRA index with the base vectors attached (make_graph_to_cagra, then attach_dataset), which
# the CAGRA sweep of Figure 7 (src/search/cagra/cagra_iter_sweep.py, cuVS 25.02 Python) searches: for every
# iteration count of experiments/qps_recall/it_schedule.txt the first query batch runs once as an untimed warm-up
# and three times timed (sweep.tsv holds the means), with itopk = max(32, it rounded up to a multiple of 32) and
# search width 1.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

DS=${DS:-sift1M}
BATCH=${BATCH:-10000}
K=${K:-10}
SEARCH_WIDTH=${SEARCH_WIDTH:-1}
PYTHON=${PYTHON:-python3}  # needs numpy
GRAPHS=${GRAPHS:-$INDEX_DIR/$DS/cagra/reverse_ab}
OUT=${OUT:-$OUT_DIR/reverse_edges}
QR=$BENCH/experiments/qps_recall
ITF=$QR/it_schedule.txt

HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/search"
GT=$OUT/search/gt.k10.ibin
[[ -f $GT ]] || "$PYTHON" "$QR/gt_top10.py" "$DATA_DIR/$DS/gt.ibin" "$GT"

for v in with_reverse no_reverse; do
  g=$GRAPHS/$v
  "$PYTHON" "$HERE/abin_to_logdeg.py" "$g/graph.abin" "$g/graph_logdeg.bin"
  "$BIN_DIR/make_graph_to_cagra" "$g/graph_logdeg.bin" "$g/index_graph_only.cagra"
  "$BIN_DIR/attach_dataset" "$g/index_graph_only.cagra" "$DATA_DIR/$DS/base.fvecs" "$g/index.cagra"
  mkdir -p "$OUT/search/$v"
  # index.cagra holds the dataset, so the sweep searches it as is (the --graph-degree and NN-descent settings
  # would only be used to rebuild a graph-only index)
  "$ENV_CAGRA/bin/python" "$BENCH/src/search/cagra/cagra_iter_sweep.py" \
    --index "$g/index.cagra" \
    --base "$DATA_DIR/$DS/base.fbin" --query "$DATA_DIR/$DS/query.fbin" --gt "$GT" \
    --out "$OUT/search/$v" --batch-size "$BATCH" --it-file "$ITF" \
    --k "$K" --itopk-size 64 --search-width "$SEARCH_WIDTH" --gpu 0 \
    --graph-degree 32 --intermediate-graph-degree 64 --nn-descent-niter 20 \
    --cache-dir "$g/_search_cache" \
    > >(tee "$OUT/search/$v/run.log") 2>&1
done
echo "sweeps in $OUT/search/{with_reverse,no_reverse}/sweep.tsv"
