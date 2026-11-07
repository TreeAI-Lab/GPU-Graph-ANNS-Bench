#!/bin/bash
# Batch-size sweep of HNSW with 20 threads (HNSW_20 in Figure 10) on SIFT1M: Faiss IndexHNSWFlat with M = 16 (maximum degree 32) and
# efConstruction = 60, searched with 20 threads at every efSearch of 10, 12, ..., 500.
# Usage: bash experiments/batch_size/run_hnsw.sh [batch size ...]   (default: 1 10 100 1000 10000)
# Needs: $BIN_DIR/faiss_hnsw (bash src/baselines/build.sh faiss)
# Output: $INDEX_DIR/sift1M/hnsw/M16_efc60/hnsw_M16.faissindex, $OUT_DIR/batch_size/hnsw/<batch>.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

BATCHES=${*:-"1 10 100 1000 10000"}
BIN=${BIN:-$BIN_DIR/faiss_hnsw}
DATA=$DATA_DIR/sift1M
INDEX=$INDEX_DIR/sift1M/hnsw/M16_efc60/hnsw_M16.faissindex
OUT=$OUT_DIR/batch_size/hnsw
THREADS=${THREADS:-20}
export LD_LIBRARY_PATH=$W/build/faiss-hnsw/install/lib:$ENV_FAISS/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$OUT" "$(dirname "$INDEX")"
[[ -f $INDEX ]] || "$BIN" build "$DATA" 16 60 10 "$INDEX" > "$OUT/build.log" 2>&1
seq 10 2 500 > "$OUT/ef_10_500_s2.txt"
for b in $BATCHES; do
  clear_caches
  "$BIN" search "$DATA" "$INDEX" 64 10 --threads "$THREADS" --batch "$b" --ef-file "$OUT/ef_10_500_s2.txt" \
    > "$OUT/$b.log" 2>&1
done
