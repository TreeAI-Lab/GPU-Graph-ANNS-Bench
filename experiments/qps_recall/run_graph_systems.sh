#!/bin/bash
# QPS-recall sweeps of CAGRA, GGNN, GANNS and SONG (Figures 1, 7 and 10).
# Usage: bash experiments/qps_recall/run_graph_systems.sh [dataset ...]  (default: all datasets of datasets.sh)
#   ONLY=<cagra|ggnn|ganns|song>  run one system;  FORCE=1  repeat sweeps that are already complete
# Every sweep searches the first batch of BATCH queries (all queries of these datasets) at every point of the
# iteration schedule: one untimed warm-up run, then three timed runs; sweep.tsv holds the means.
# Output: $OUT_DIR/qps_recall/<dataset>/<system>/ (sweep.tsv, it_batches.tsv, gpu_mem_cuda.tsv, load_timing.tsv,
#   params.txt, run.log) and $OUT_DIR/qps_recall/<dataset>/gt.k10.ibin
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/datasets.sh"

ONLY=${ONLY:-}
FORCE=${FORCE:-0}
BATCH=${BATCH:-10000}
ITF=$HERE/it_schedule.txt          # CAGRA, GGNN, GANNS: it 15..500, 68 points
ITS=$HERE/it_schedule_song.txt     # SONG: it 15..240, 59 points
OUT=$OUT_DIR/qps_recall
PY_CAGRA=$ENV_CAGRA/bin/python
mkdir -p "$OUT"

maybe() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

# Top-10 slice of the k = 100 ground truth
prepare_gt() {
  local ds=$1 data=$2
  local dst=$OUT/$ds/gt.k10.ibin
  mkdir -p "$OUT/$ds"
  [[ -f $dst ]] || "$PY_CAGRA" "$HERE/gt_top10.py" "$DATA_DIR/$data/gt.ibin" "$dst" >&2
  echo "$dst"
}

run_one() {
  local ds=$1 algo=$2
  local data n dim deg nq ctag gtag atag stag shift gt out need nrows rc
  read -r data n dim deg nq ctag gtag atag stag shift <<<"$(cfg_graph "$ds")"
  case $algo in
    ganns) [[ $atag == - ]] && return 0 ;;
    song) [[ $stag == - ]] && return 0 ;;
  esac
  gt=$(prepare_gt "$ds" "$data")
  local D=$DATA_DIR/$data G=$INDEX_DIR/$data
  out=$OUT/$ds/$algo
  need=68
  [[ $algo == song ]] && need=59
  if [[ -f $out/sweep.tsv && $FORCE != 1 ]]; then
    nrows=$(awk 'NR>1{c++} END{print c+0}' "$out/sweep.tsv")
    if [[ $nrows -ge $need ]]; then
      echo "SKIP complete $ds $algo ($nrows rows)"
      return 0
    fi
  fi
  mkdir -p "$out"
  echo "======== $ds $algo $(date '+%F %T') ========"
  set +e
  case "$algo:$ds" in
    cagra:deep100M)
      # Graph-only index from the IVF-PQ builder; the C++ driver attaches the base in-process
      "$BIN_DIR/cagra_resident_search" \
        --index "$G/cagra/$ctag/index.cagra" \
        --base "$D/base.fbin" --query "$D/query.fbin" --gt "$gt" \
        --out "$out" --batch-size "$BATCH" --k 10 \
        --search-width 1 --it-file "$ITF" --gpu 0 \
        > >(tee "$out/run.log") 2>&1
      ;;
    cagra:*)
      # The index is graph-only: the first run rebuilds it with the dataset and caches it (see cagra_py_common.py)
      "$PY_CAGRA" "$BENCH/src/search/cagra/cagra_iter_sweep.py" \
        --index "$G/cagra/$ctag/index.cagra" \
        --base "$D/base.fbin" --query "$D/query.fbin" --gt "$gt" \
        --out "$out" --batch-size "$BATCH" --it-file "$ITF" \
        --k 10 --itopk-size 64 --search-width 1 --gpu 0 \
        --graph-degree "$deg" --intermediate-graph-degree "$((deg * 2))" --nn-descent-niter 20 \
        --cache-dir "$G/cagra/$ctag/_search_cache" \
        > >(tee "$out/run.log") 2>&1
      ;;
    ggnn:deep100M)
      "$BIN_DIR/ggnn_resident_search" \
        --base "$D/base.fbin" --graph-dir "$G/ggnn/$gtag" \
        --query "$D/query.fbin" --gt "$gt" --out "$out" \
        --k-build "$deg" --k-query 10 --tau-query 0.5 \
        --batch-size "$BATCH" --gpu 0 --it-file "$ITF" \
        --shard-size 20000000 \
        > >(tee "$out/run.log") 2>&1
      ;;
    ggnn:*)
      "$BIN_DIR/ggnn_resident_search" \
        --base "$D/base.fvecs" --graph-dir "$G/ggnn/$gtag" \
        --query "$D/query.fbin" --gt "$gt" --out "$out" \
        --k-build "$deg" --k-query 10 --tau-query 0.5 \
        --batch-size "$BATCH" --gpu 0 --it-file "$ITF" \
        > >(tee "$out/run.log") 2>&1
      ;;
    ganns:*)
      "$BIN_DIR/ganns_resident_search_d$dim" \
        --base "$D/base.fvecs" --graph "$G/ganns/$atag/graph.nsw" \
        --query "$D/query.fbin" --gt "$gt" --out "$out" \
        --k 10 --M 512 --n-e 15 --batch-size "$BATCH" --gpu 0 --it-file "$ITF" \
        > >(tee "$out/run.log") 2>&1
      ;;
    song:*)
      "$BIN_DIR/song_resident_search_d${dim}_fd${deg}" \
        --base "$D/base.fbin" --graph "$G/song/$stag/bfsg.graph" \
        --query "$D/query.fbin" --gt "$gt" --out "$out" \
        --batch-size "$BATCH" --gpu 0 --n-base "$n" --dim "$dim" --shift "$shift" \
        --max-iterations 500 --kernel-chunk 0 --it-file "$ITS" \
        > >(tee "$out/run.log") 2>&1
      ;;
  esac
  rc=$?
  set -e
  {
    echo "dataset	$ds"
    echo "algo	$algo"
    echo "batch_size	$BATCH"
    echo "max_degree	$deg"
    echo "query	$D/query.fbin"
    echo "gt	$gt"
    echo "exit_code	$rc"
    echo "finished_at	$(date '+%F %T')"
  } >>"$out/params.txt"
  if [[ $rc -ne 0 ]]; then
    echo "FAIL $ds $algo rc=$rc" >&2
    return $rc
  fi
}

DATASETS=${*:-$GRAPH_DATASETS}
fail=0
for ds in $DATASETS; do
  maybe cagra && { run_one "$ds" cagra || fail=1; }
  maybe ggnn  && { run_one "$ds" ggnn  || fail=1; }
  maybe ganns && { run_one "$ds" ganns || fail=1; }
  maybe song  && { run_one "$ds" song  || fail=1; }
done
echo "===== graph systems done $(date '+%F %T') fail=$fail ====="
exit $fail
