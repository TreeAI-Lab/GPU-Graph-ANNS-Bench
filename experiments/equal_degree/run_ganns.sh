#!/bin/bash
# GANNS search of the equal-degree runs (Figures 6(b), 8, and 13; Table 4 for the average degree and
# the connected components, which the query program prints): per candidate list size e the query program runs
# once as an untimed warm-up and then TIMED_RUNS times (default 3; repeat_runs in common.sh, one process per
# run). The query program computes Recall@K itself against the ground truth in ivecs format. The log of a setting holds
# all runs after marker lines.
#
# Usage:  bash run_ganns.sh [dataset ...]     (default: the datasets listed below)
# Needs:  build_ganns.sh <dim> <metric>, build_indexes.sh ganns <dataset> [degree]
# Output: $OUT_DIR/equal_degree/ganns/<run>/k<K>/e<e>.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

E_LIST=${E_LIST:-$(seq 10 2 500)}   # candidate list sizes
K=${K:-10}
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(gist deep10M sift1M nytimes glove200 mnist8M deep1M)

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  case $ds in glove200|nytimes) metric=cos ;; *) metric=l2 ;; esac
  run=$ds; [ -z "${DEGREE:-}" ] || run=$ds-d$deg
  query_bin=$EQDEG_BIN/ganns/GANNS_$(dim_of "$ds")_$metric
  graph=$INDEX_DIR/$ds/ganns/eqdeg-d$deg/graph.nsw
  gt=$(gt_ivecs "$ds")
  out=$EQDEG_OUT/ganns/$run/k$K
  write_meta "$out" ganns "$ds" "$K"
  for e in $E_LIST; do
    echo "Running GANNS on $ds with e=$e ..."
    repeat_runs "$out/e$e.log" : \
      "$query_bin" -s "$DATA_DIR/$ds/base.fvecs" "$DATA_DIR/$ds/query.fvecs" "$graph" "$gt" "$e" "$K"
  done
done

echo "All GANNS experiments finished."
