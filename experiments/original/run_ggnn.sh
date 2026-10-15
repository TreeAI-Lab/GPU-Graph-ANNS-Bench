#!/bin/bash
# GGNN search of the original runs (Figures 5(a), 7, 9, 11, 12): per max_iterations value the original driver
# runs its query once as an untimed warm-up and then TIMED_RUNS times (default 3; repeat_runs in common.sh,
# one process per run), and Recall@K is computed from the result file of every timed run. The log of a setting holds all
# runs after marker lines.
#
# Usage:  bash run_ggnn.sh [dataset ...]      (default: the datasets of the original run script)
# Needs:  build_ggnn.sh, build_indexes.sh ggnn <dataset> [degree]
# Output: $OUT_DIR/original/ggnn/<run>/k<K>/it<it>.log, it<it>.txt (result ids of the last timed run) and
#         it<it>.recall (one line per timed run)
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

IT_LIST=${IT_LIST:-$(seq 30 2 60)}   # max_iterations values
K=${K:-10}                           # KQuery and K of Recall@K
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(gist sift1M nytimes-full deep10M glove200 mnist8M deep100M deep1M)

# distance of every dataset in the original run script
metric_of() {
  case $1 in nytimes-full|nytimes|deep10M|glove200) echo cos ;; *) echo l2 ;; esac
}

# Recall@K of the result file of one timed run, appended to it<it>.recall
ggnn_recall() {
  python3 "$ORIG_TOOLS/cal_recall.py" --gt "$DATA_DIR/$ds/gt.ibin" --ret "$out/it$it.txt" --k "$K" \
    --out "$out/it$it.recall" --append
}

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  run=$ds; [ -z "${DEGREE:-}" ] || run=$ds-d$deg
  graph=$INDEX_DIR/$ds/ggnn/orig-d$deg
  out=$ORIG_OUT/ggnn/$run/k$K
  write_meta "$out" ggnn "$ds" "$K"
  for it in $IT_LIST; do
    echo "Running GGNN on $ds with max_iterations=$it ..."
    rm -f "$out/it$it.recall"
    repeat_runs "$out/it$it.log" ggnn_recall \
      "$ORIG_BIN/ggnn_main_gpu_data" query "$DATA_DIR/$ds/base.fvecs" "$DATA_DIR/$ds/query.fvecs" "$graph" "$deg" "$K" \
      "$it" "$(metric_of "$ds")" "$out/it$it.txt"
  done
done

echo "All GGNN experiments finished."
