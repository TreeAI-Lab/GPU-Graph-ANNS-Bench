#!/bin/bash
# BANG search of the original runs (Figures 5(a), 7, 9, 11, 12): bang_search in its automatic mode, which
# raises the worklist length L by a constant step. For every L, bang_search itself searches all queries once as
# an untimed warm-up and then three times, and prints a line per run and a summary line with the mean wall-clock
# time of the three timed searches, the QPS (queries / mean time) and Recall@K (third_party/patches/bang.patch).
# With the timers of the patched build every search also prints the per-stage times (STATS) and the counters
# calc_count and path_length; tools/parse_logs.py averages those of the timed searches.
# BANG treats every distance name except mips as L2. With L_LIST set, bang_search runs once per L with that
# fixed worklist length instead (L<L>.log), with the same warm-up and timed searches.
#
# Usage:  bash run_bang.sh [dataset ...]      (default: the datasets of the original run script)
# Needs:  third_party/build/bang.sh <degree>, build_indexes.sh vamana <dataset> [degree]
# Output: $OUT_DIR/original/bang/<run>/k<K>/auto.log, or L<L>.log per value of L_LIST
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

K=${K:-10}
L_LIST=${L_LIST:-}
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(gist sift1M nytimes-full deep10M glove200 mnist8M deep100M deep1M)

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  case $ds in nytimes-full|nytimes|glove200) dist=cos ;; *) dist=l2 ;; esac
  run=$ds; [ -z "${DEGREE:-}" ] || run=$ds-d$deg
  prefix=$INDEX_DIR/$ds/vamana/orig-d$deg/vamana
  out=$ORIG_OUT/bang/$run/k$K
  write_meta "$out" bang "$ds" "$K"
  echo "Running BANG on $ds ..."
  bang=("$W/build/bang/r$deg/bang_search" "$prefix" "$DATA_DIR/$ds/query.fbin" "$DATA_DIR/$ds/gt.ibin" "$(nq_of "$ds")"
        "$K" float "$dist")
  if [ -z "$L_LIST" ]; then
    "${bang[@]}" > "$out/auto.log" 2>&1
  else
    for L in $L_LIST; do "${bang[@]}" "$L" > "$out/L$L.log" 2>&1; done
  fi
done

echo "All BANG experiments finished."
