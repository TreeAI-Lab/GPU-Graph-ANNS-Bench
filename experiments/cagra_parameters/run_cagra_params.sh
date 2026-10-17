#!/bin/bash
# Figure 13: CAGRA on SIFT1M with one search parameter varied at a time from the base setting
# (search width 1, internal top-k 32, graph degree 32, team size 8, 40 iterations, k = 10).
# Every setting searches the 10,000 queries as one batch with the CAGRA sweep of Figure 7
# (src/search/cagra/cagra_iter_sweep.py): one untimed warm-up run, then three timed runs, means reported.
# The degree runs search CAGRA graphs of degree D built as in Figure 7 (intermediate degree 2D, 20 NN-descent
# iterations, src/build/cagra_build.py); a missing graph is built first. Degree 32 is the graph of Figure 7.
#
# Usage:  bash experiments/cagra_parameters/run_cagra_params.sh       FORCE=1 repeats finished settings
# Output: $INDEX_DIR/sift1M/cagra/gd<D>_igd<2D>_it20/index.cagra
#         $OUT_DIR/cagra_parameters/<parameter>-<value>/ (sweep.tsv, it_batches.tsv, params.txt, run.log, ...)
#         then: python3 experiments/cagra_parameters/summarize.py --root $OUT_DIR/cagra_parameters
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

FORCE=${FORCE:-0}
IT=${IT:-40}
BASE_SEARCH_WIDTH=${BASE_SEARCH_WIDTH:-1}
BASE_ITOPK=${BASE_ITOPK:-32}
BASE_DEGREE=${BASE_DEGREE:-32}
BASE_TEAM_SIZE=${BASE_TEAM_SIZE:-8}
SEARCH_WIDTHS=${SEARCH_WIDTHS:-"1 2 4 8 16 32"}
ITOPKS=${ITOPKS:-"16 32 64 128 256"}
DEGREES=${DEGREES:-"16 32 64 96 128"}
TEAM_SIZES=${TEAM_SIZES:-"8 16 32"}

DATA=$DATA_DIR/sift1M
G=$INDEX_DIR/sift1M/cagra
OUT=$OUT_DIR/cagra_parameters
PY=$ENV_CAGRA/bin/python
mkdir -p "$OUT"
printf '%s\n' "$IT" >"$OUT/it.txt"

graph() {  # graph <degree>: build the CAGRA graph of this degree if it is missing
  local deg=$1 dir
  dir=$G/gd${deg}_igd$((deg * 2))_it20
  [[ -f $dir/index.cagra ]] && return 0
  mkdir -p "$dir"
  echo "build CAGRA sift1M degree $deg"
  "$PY" "$BENCH/src/build/cagra_build.py" "$DATA/base.fbin" "$dir/index.cagra" "$deg" "$((deg * 2))" 20 \
    >"$dir/build.log" 2>&1
}

run() {  # run <parameter> <value> <degree> <search width> <internal top-k> <team size>
  local out=$OUT/$1-$2 deg=$3 dir
  dir=$G/gd${deg}_igd$((deg * 2))_it20
  if [[ -f $out/sweep.tsv && $FORCE != 1 ]]; then
    echo "SKIP $1-$2 (done)"
    return 0
  fi
  graph "$deg"
  mkdir -p "$out"
  "$PY" "$BENCH/src/search/cagra/cagra_iter_sweep.py" \
    --index "$dir/index.cagra" --base "$DATA/base.fbin" --query "$DATA/query.fbin" --gt "$DATA/gt.ibin" \
    --out "$out" --batch-size 10000 --it-file "$OUT/it.txt" --k 10 \
    --search-width "$4" --fixed-itopk "$5" --team-size "$6" --gpu 0 \
    --graph-degree "$deg" --intermediate-graph-degree "$((deg * 2))" --nn-descent-niter 20 \
    --cache-dir "$dir/_search_cache" \
    > >(tee "$out/run.log") 2>&1
}

for v in $SEARCH_WIDTHS; do run search_width "$v" "$BASE_DEGREE" "$v" "$BASE_ITOPK" "$BASE_TEAM_SIZE"; done
for v in $ITOPKS; do run internal_topk "$v" "$BASE_DEGREE" "$BASE_SEARCH_WIDTH" "$v" "$BASE_TEAM_SIZE"; done
for v in $DEGREES; do run degree "$v" "$v" "$BASE_SEARCH_WIDTH" "$BASE_ITOPK" "$BASE_TEAM_SIZE"; done
for v in $TEAM_SIZES; do run team_size "$v" "$BASE_DEGREE" "$BASE_SEARCH_WIDTH" "$BASE_ITOPK" "$v"; done

echo "CAGRA parameter runs in $OUT"
