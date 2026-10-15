#!/bin/bash
# PilotANN search of the original runs (Figures 7, 11): the PilotANN profiler on 1,024 random queries, drawn
# once per process, one process per ef_search value. tools/pilotann_bench.py searches these queries once as an
# untimed warm-up and then three times, and prints the mean recall and the mean time of the three timed searches.
#
# Usage:  bash run_pilotann.sh [dataset ...]  (default: the datasets of the original run script)
# Needs:  third_party/build/pilotann.sh, build_indexes.sh pilotann <dataset> [degree]
# Output: $OUT_DIR/original/pilotann/<run>/k<K>/ef<ef>.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

K=${K:-10}
ALGORITHM=${ALGORITHM:-nsg}
N_QUERIES=${N_QUERIES:-1024}
EF_LIST=${EF_LIST:-$(seq 30 2 80)}
PY=$ENV_PILOT/bin/python
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(deep10M sift1M gist deep1M mnist8M)

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  run=$ds; [ -z "${DEGREE:-}" ] || run=$ds-d$deg
  ckpt=$INDEX_DIR/$ds/pilotann/orig-d$deg/$ds.ckpt
  data=$ORIG_OUT/pilotann_data/$ds
  mkdir -p "$data"
  ln -sf "$DATA_DIR/$ds/base.fvecs" "$data/base.fvecs"
  ln -sf "$DATA_DIR/$ds/query.fvecs" "$data/query.fvecs"
  ln -sf "$(gt_ivecs "$ds")" "$data/groundtruth.ivecs"
  out=$ORIG_OUT/pilotann/$run/k$K
  write_meta "$out" pilotann "$ds" "$K" "$N_QUERIES"

  # release cached GPU memory and wait before the next dataset
  "$PY" -c "
import torch, time
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
time.sleep(10)
"
  for ef in $EF_LIST; do
    echo "Running checkpoint=$ckpt ef_search=$ef ..."
    "$PY" "$ORIG_TOOLS/pilotann_bench.py" --dataset_dir "$data" --checkpoint "$ckpt" --top_k "$K" \
      --algorithm "$ALGORITHM" --n_queries "$N_QUERIES" --ef_search "[$ef]" > "$out/ef$ef.log" 2>&1
  done
done

echo "All searches finished."
