#!/bin/bash
# Figure 12, CAGRA curves: SIFT1M with maximum degree 64 and 128.
# Usage:  bash experiments/topk_degree/run_cagra_degree.sh
# Needs:  $ENV_CAGRA/bin/python, src/build/cagra_build.py, src/search/cagra/cagra_iter_sweep.py
# Output: $OUT_DIR/topk_degree/cagra/degree<D>/
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

DEGREES=${DEGREES:-64 128}
IT_LIST=${IT_LIST:-$(seq 10 2 300)}
SEARCH_WIDTH=${SEARCH_WIDTH:-1}
ITOPK=${ITOPK:-128}
TEAM_SIZE=${TEAM_SIZE:-8}
DATA=$DATA_DIR/sift1M
PY=$ENV_CAGRA/bin/python

for D in $DEGREES; do
  dir=$INDEX_DIR/sift1M/cagra/gd${D}_igd$((D * 2))_it20
  out=$OUT_DIR/topk_degree/cagra/degree$D
  mkdir -p "$dir" "$out"
  if [[ ! -f $dir/index.cagra ]]; then
    "$PY" "$BENCH/src/build/cagra_build.py" "$DATA/base.fbin" "$dir/index.cagra" "$D" "$((D * 2))" 20 \
      > "$dir/build.log" 2>&1
  fi
  printf '%s\n' $IT_LIST > "$out/it.txt"
  "$PY" "$BENCH/src/search/cagra/cagra_iter_sweep.py" \
    --index "$dir/index.cagra" --base "$DATA/base.fbin" --query "$DATA/query.fbin" --gt "$DATA/gt.ibin" \
    --out "$out" --batch-size 10000 --it-file "$out/it.txt" --k 10 \
    --search-width "$SEARCH_WIDTH" --fixed-itopk "$ITOPK" --team-size "$TEAM_SIZE" --gpu 0 \
    --graph-degree "$D" --intermediate-graph-degree "$((D * 2))" --nn-descent-niter 20 \
    --cache-dir "$dir/_search_cache" \
    > "$out/run.log" 2>&1
done
