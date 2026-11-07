#!/bin/bash
# One-time load of Figure 9: host-to-device copy time of the base vectors and of each system's graph,
# measured by $BIN_DIR/h2d_copy_bench (pageable host buffer, WARMUP untimed and REPS timed cudaMemcpy
# calls; the figure uses mean_ms). Each system's graph is copied on deep1M and deep10M; deep100M
# copies the base vectors and the CAGRA graph.
#
# Usage: bash experiments/loading/run_h2d_bench.sh [datasets...]   (default: deep1M deep10M deep100M)
# Output: $OUT_DIR/loading/h2d/summary.tsv with columns
#   kind dataset algorithm config file bytes median_ms min_ms max_ms mean_ms median_GBps trials_ms
#   (kind = dataset for the base vectors, graph for that system's index)
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/../qps_recall/datasets.sh"

BIN=${BIN:-$BIN_DIR/h2d_copy_bench}
WARMUP=${WARMUP:-2}
REPS=${REPS:-5}
OUT=${OUT:-$OUT_DIR/loading/h2d}
if [[ $# -gt 0 ]]; then DATASETS=("$@"); else DATASETS=(deep1M deep10M deep100M); fi

# Base file copied for each dataset: the fvecs file, except the fbin file for deep100M.
base_file() {
  case "$1" in
    deep100M) echo "$DATA_DIR/$1/base.fbin" ;;
    *) echo "$DATA_DIR/$1/base.fvecs" ;;
  esac
}

copy_file() {
  local kind=$1 ds=$2 algo=$3 config=$4 file=$5 line
  echo "=== $ds $algo $file"
  line=$("$BIN" "$ds" "$algo" "$config" "$file" "$WARMUP" "$REPS" 2>>"$OUT/run.log")
  printf '%s\t%s\n' "$kind" "$line" | tee -a "$SUMMARY"
}

[ -x "$BIN" ] || { echo "missing $BIN (bash src/bench/build.sh)"; exit 2; }
mkdir -p "$OUT"
SUMMARY=$OUT/summary.tsv
echo -e "kind\tdataset\talgorithm\tconfig\tfile\tbytes\tmedian_ms\tmin_ms\tmax_ms\tmean_ms\tmedian_GBps\ttrials_ms" >"$SUMMARY"

for ds in "${DATASETS[@]}"; do
  read -r data n dim deg nq ctag gtag atag stag shift <<<"$(cfg_graph "$ds")"
  copy_file dataset "$ds" base base "$(base_file "$ds")"
  copy_file graph "$ds" cagra "$ctag" "$INDEX_DIR/$ds/cagra/$ctag/index.cagra"
  if [[ $ds == deep100M ]]; then
    continue
  fi
  ggnn=$INDEX_DIR/$ds/ggnn/$gtag
  parts=("$ggnn"/part_*.ggnn)
  [[ -f ${parts[0]} ]] || { echo "missing GGNN graph $ggnn" >&2; exit 2; }
  for part in "${parts[@]}"; do
    copy_file graph "$ds" ggnn "$gtag" "$part"
  done
  copy_file graph "$ds" ganns "$atag" "$INDEX_DIR/$ds/ganns/$atag/graph.nsw"
  copy_file graph "$ds" song "$stag" "$INDEX_DIR/$ds/song/$stag/bfsg.graph"
done
echo "wrote $SUMMARY"
