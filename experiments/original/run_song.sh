#!/bin/bash
# SONG search of the original runs (Figures 5(a), 7, 9, 11, 12): per value of max_iterations (the queue holds
# max_iterations - 1 entries) the SONG program runs once as an untimed warm-up and then TIMED_RUNS times
# (default 3; repeat_runs in common.sh, one process per run), and Recall@K is computed from the output of every timed run.
# stdout and stderr of a run go to one file, as in the original run script; cal_recall.py skips the 7 lines
# before and the 2 lines after the result lines. The log of a setting holds all runs after marker lines.
#
# Usage:  bash run_song.sh [dataset ...]      (default: the datasets of the original run script)
# Needs:  build_song.sh <dataset> [degree], build_indexes.sh song <dataset> [degree]
# Output: $OUT_DIR/original/song/<run>/k<K>/it<it>.log and it<it>.recall (one line per timed run); <run> is the
#         dataset, or <dataset>-d<degree> when DEGREE is set
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

IT_LIST=${IT_LIST:-$(seq 30 2 60)}   # max_iterations values
K=${K:-10}                           # results printed per query and K of Recall@K
SEARCH_TOP_K=100                     # used only by the CPU program
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(nytimes-full sift1M gist glove200 deep1M deep10M)

# Recall@K of one timed run, read from its output, appended to it<it>.recall
song_recall() {
  python3 "$ORIG_TOOLS/cal_recall.py" --gt "$DATA_DIR/$ds/gt.ibin" --ret "$1" --k "$K" \
    --skip-head 7 --skip-tail 2 --out "$out/it$it.recall" --append
}

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  case $ds in nytimes-full|nytimes|glove200) metric=cos ;; *) metric=l2 ;; esac
  run=$ds; [ -z "${DEGREE:-}" ] || run=$ds-d$deg
  bin=$ORIG_BIN/song/$ds-d$deg/song
  graph=$INDEX_DIR/$ds/song/orig-d$deg
  out=$ORIG_OUT/song/$run/k$K
  write_meta "$out" song "$ds" "$K"
  echo "Running SONG on $ds ..."
  for it in $IT_LIST; do
    rm -f "$out/it$it.recall"
    repeat_runs "$out/it$it.log" song_recall \
      "$bin" test 0 "$DATA_DIR/$ds/query.fvecs" "$SEARCH_TOP_K" "$(nb_of "$ds")" "$(dim_of "$ds")" "$K" "$metric" \
      "$it" "$graph"
  done
done

echo "All SONG experiments finished."
