#!/bin/bash
# PilotANN search of the equal-degree runs (Figures 6(b) and 8): the PilotANN profiler on every query of the dataset,
# in file order, one process per ef_search value. tools/pilotann_bench.py searches these queries once as an
# untimed warm-up and then three times, and prints the mean recall and the mean time of the three timed searches.
#
# Usage:  bash run_pilotann.sh [dataset ...]  (default: the datasets listed below)
# Needs:  third_party/build/pilotann.sh, build_indexes.sh pilotann <dataset> [degree]
# Output: $OUT_DIR/equal_degree/pilotann/<run>/k<K>/ef<ef>.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

K=${K:-10}
ALGORITHM=${ALGORITHM:-nsg}
EF_LIST=${EF_LIST:-$(seq 10 2 500)}
PY=$ENV_PILOT/bin/python
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(deep10M sift1M gist deep1M mnist8M)

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  run=$ds; [ -z "${DEGREE:-}" ] || run=$ds-d$deg
  ckpt=$INDEX_DIR/$ds/pilotann/eqdeg-d$deg/$ds.ckpt
  data=$EQDEG_OUT/pilotann_data/$ds
  mkdir -p "$data"
  ln -sf "$DATA_DIR/$ds/base.fvecs" "$data/base.fvecs"
  ln -sf "$DATA_DIR/$ds/query.fvecs" "$data/query.fvecs"
  ln -sf "$(gt_ivecs "$ds")" "$data/groundtruth.ivecs"
  out=$EQDEG_OUT/pilotann/$run/k$K
  nq=${N_QUERIES:-$(nq_of "$ds")}
  write_meta "$out" pilotann "$ds" "$K" "$nq"

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
    "$PY" "$EQDEG_TOOLS/pilotann_bench.py" --dataset_dir "$data" --checkpoint "$ckpt" --top_k "$K" \
      --algorithm "$ALGORITHM" --n_queries "$nq" --ef_search "[$ef]" > "$out/ef$ef.log" 2>&1
  done
done

echo "All searches finished."
