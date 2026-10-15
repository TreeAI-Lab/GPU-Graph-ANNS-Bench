#!/bin/bash
# Figure 16, step 2: share of short and long edges of the two graphs.
# sample_top32 draws 10,000 base vectors and finds their exact 32 nearest neighbors (cuVS brute force);
# edge_length_stats labels each graph edge of a sampled node short if its target is among those 32, long otherwise.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

DS=${DS:-sift1M}
SAMPLES=${SAMPLES:-10000}  # sampled nodes
K=${K:-32}                 # exact neighbors per sampled node
SEED=${SEED:-42}
BF_INDEX=${BF_INDEX:-}     # optional serialized cuVS brute-force index of the base set; built from base.fbin if empty
GRAPHS=${GRAPHS:-$INDEX_DIR/$DS/cagra/reverse_ab}
OUT=${OUT:-$OUT_DIR/reverse_edges}

mkdir -p "$OUT/sample" "$OUT/edge_stats"
"$BIN_DIR/sample_top32" "$DATA_DIR/$DS" "$OUT/sample" "$SAMPLES" "$K" "$SEED" ${BF_INDEX:+"$BF_INDEX"} \
  2>&1 | tee "$OUT/sample/run.log"
"$BIN_DIR/edge_length_stats" "$DATA_DIR/$DS" "$OUT/sample" "$OUT/edge_stats" \
  with_reverse="$GRAPHS/with_reverse/graph.csr" \
  no_reverse="$GRAPHS/no_reverse/graph.csr" \
  2>&1 | tee "$OUT/edge_stats/run.log"
cat "$OUT/edge_stats/summary.tsv"
