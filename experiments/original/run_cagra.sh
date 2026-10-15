#!/bin/bash
# CAGRA search of the original runs (Figures 7, 12): my_cagra search with search width 1, internal top-k
# 128 and team size 8 for every value of max_iterations, then Recall@10 of the result file (my_cagra
# returns k = 10). my_cagra prints no search time.
#
# Usage:  bash run_cagra.sh [dataset ...]     (default: the datasets of the original run script)
# Needs:  build_cagra.sh, build_indexes.sh cagra <dataset> [degree]
# Output: $OUT_DIR/original/cagra/<run>/k10/it<it>.log, it<it>.topk and it<it>.recall
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

IT_LIST=${IT_LIST:-$(seq 30 2 60)}
SEARCH_WIDTH=${SEARCH_WIDTH:-1}
ITOPK=${ITOPK:-128}
TEAM_SIZE=${TEAM_SIZE:-8}
K=10
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(deep10M glove200 nytimes-full sift1M mnist8M deep100M deep1M gist)

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  run=$ds; [ -z "${DEGREE:-}" ] || run=$ds-d$deg
  index=$INDEX_DIR/$ds/cagra/orig-d$deg/index
  out=$ORIG_OUT/cagra/$run/k$K
  write_meta "$out" cagra "$ds" "$K"
  for it in $IT_LIST; do
    echo "Running CAGRA on $ds with iter=$it ..."
    "$ORIG_BIN/my_cagra" search "$index" "$DATA_DIR/$ds/query.fvecs" "$it" "$out/it$it.topk" \
      "$SEARCH_WIDTH" "$ITOPK" "$TEAM_SIZE" > "$out/it$it.log" 2>&1
    python3 "$ORIG_TOOLS/cal_recall.py" --gt "$DATA_DIR/$ds/gt.ibin" --ret "$out/it$it.topk" --k "$K" \
      --skip-head 3 --out "$out/it$it.recall"
  done
done

echo "All CAGRA experiments finished."
