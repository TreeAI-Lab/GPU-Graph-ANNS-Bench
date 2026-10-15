#!/bin/bash
# GANNS search of the original runs (Figures 5(a), 7, 9, 11, 12; Tables 5 and 6 for the average degree and
# the connected components, which the query program prints): per candidate list size e the query program runs
# once as an untimed warm-up and then TIMED_RUNS times (default 3; repeat_runs in common.sh, one process per
# run). The
# query program computes Recall@K itself against the ground truth in ivecs format. The log of a setting holds
# all runs after marker lines.
#
# Usage:  bash run_ganns.sh [dataset ...]     (default: the datasets of the original run script)
# Needs:  build_ganns.sh <dim> <metric>, build_indexes.sh ganns <dataset> [degree]
# Output: $OUT_DIR/original/ganns/<run>/k<K>/e<e>.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

E_LIST=${E_LIST:-$(seq 30 5 60)}   # candidate list sizes
K=${K:-10}
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(gist deep10M sift1M nytimes-full glove200 mnist8M deep1M)

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  case $ds in nytimes-full|nytimes|glove200) metric=cos ;; *) metric=l2 ;; esac
  run=$ds; [ -z "${DEGREE:-}" ] || run=$ds-d$deg
  query_bin=$ORIG_BIN/ganns/query_$(dim_of "$ds")_$metric
  graph=$INDEX_DIR/$ds/ganns/orig-d$deg/graph.nsw
  gt=$(gt_ivecs "$ds")
  out=$ORIG_OUT/ganns/$run/k$K
  write_meta "$out" ganns "$ds" "$K"
  for e in $E_LIST; do
    echo "Running GANNS on $ds with e=$e ..."
    repeat_runs "$out/e$e.log" : \
      "$query_bin" "$DATA_DIR/$ds/base.fvecs" "$DATA_DIR/$ds/query.fvecs" nsw "$graph" "$gt" "$e" "$K"
  done
done

echo "All GANNS experiments finished."
