#!/bin/bash
# One-time load of Figure 6: host-to-device copy time of the base vectors and of the CAGRA graph of
# each dataset, measured by $BIN_DIR/h2d_copy_bench (pageable host buffer, WARMUP untimed and REPS
# timed cudaMemcpy calls; the figure uses mean_ms).
#
# Usage: bash experiments/loading/run_h2d_bench.sh [datasets...]   (default: deep1M deep10M-annb deep100M)
# Output: $OUT_DIR/loading/h2d/summary.tsv with columns
#   kind dataset algorithm config file bytes median_ms min_ms max_ms mean_ms median_GBps trials_ms
#   (kind = dataset for the base vectors, graph for the CAGRA index)
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

BIN=${BIN:-$BIN_DIR/h2d_copy_bench}
WARMUP=${WARMUP:-2}
REPS=${REPS:-5}
OUT=${OUT:-$OUT_DIR/loading/h2d}
if [[ $# -gt 0 ]]; then DATASETS=("$@"); else DATASETS=(deep1M deep10M-annb deep100M); fi

# Base file copied for each dataset: the fvecs file, except the fbin file for deep100M.
base_file() {
  case "$1" in
    deep100M) echo "$DATA_DIR/$1/base.fbin" ;;
    *) echo "$DATA_DIR/$1/base.fvecs" ;;
  esac
}

# CAGRA graph used by the serving runs of each dataset
cagra_tag() {
  case "$1" in
    deep100M) echo gd32_igd64_ivfpq ;;
    *) echo gd32_igd64_it20 ;;
  esac
}

[ -x "$BIN" ] || { echo "missing $BIN (bash src/bench/build.sh)"; exit 2; }
mkdir -p "$OUT"
SUMMARY=$OUT/summary.tsv
echo -e "kind\tdataset\talgorithm\tconfig\tfile\tbytes\tmedian_ms\tmin_ms\tmax_ms\tmean_ms\tmedian_GBps\ttrials_ms" >"$SUMMARY"

for ds in "${DATASETS[@]}"; do
  base=$(base_file "$ds")
  tag=$(cagra_tag "$ds")
  graph=$INDEX_DIR/$ds/cagra/$tag/index.cagra
  echo "=== $ds base $base"
  line=$("$BIN" "$ds" base base "$base" "$WARMUP" "$REPS" 2>>"$OUT/run.log")
  printf 'dataset\t%s\n' "$line" | tee -a "$SUMMARY"
  echo "=== $ds cagra $graph"
  line=$("$BIN" "$ds" cagra "$tag" "$graph" "$WARMUP" "$REPS" 2>>"$OUT/run.log")
  printf 'graph\t%s\n' "$line" | tee -a "$SUMMARY"
done
echo "wrote $SUMMARY"
