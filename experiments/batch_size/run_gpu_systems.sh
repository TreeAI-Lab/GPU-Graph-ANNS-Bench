#!/bin/bash
# Batch-size sweeps of CAGRA, GGNN, GANNS and SONG on SIFT1M (Figure 8).
# Usage: bash experiments/batch_size/run_gpu_systems.sh [batch size ...]   (default: 1 10 100 1000 10000)
# For every batch size the drivers search the first <batch> queries at every point of the iteration schedule:
# one untimed warm-up run, then three timed runs, means in sweep.tsv (the same drivers and schedules as the
# QPS-recall runs).
# Output: $OUT_DIR/batch_size/<batch>/<system>/ (sweep.tsv, it_batches.tsv, gpu_mem_cuda.tsv, load_timing.tsv,
#   params.txt, run.log); run.log of the whole run in $OUT_DIR/batch_size/run.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
QR=$BENCH/experiments/qps_recall
source "$QR/datasets.sh"

BATCHES=${*:-"1 10 100 1000 10000"}
ITF=$QR/it_schedule.txt
ITS=$QR/it_schedule_song.txt
OUT=$OUT_DIR/batch_size
PY_CAGRA=$ENV_CAGRA/bin/python
LOG=$OUT/run.log
mkdir -p "$OUT"

read -r data n dim deg nq ctag gtag atag stag shift <<<"$(cfg_graph sift1M)"
D=$DATA_DIR/$data
G=$INDEX_DIR/$data
gt=$OUT/gt.k10.ibin
[[ -f $gt ]] || "$PY_CAGRA" "$QR/gt_top10.py" "$D/gt.ibin" "$gt"
fail=0

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

run_one() {
  local algo=$1 bs=$2
  local out=$OUT/$bs/$algo
  local rc nrows
  mkdir -p "$out"
  log "START sift1M $algo bs=$bs"
  set +e
  case "$algo" in
    cagra)
      "$PY_CAGRA" "$BENCH/src/search/cagra/cagra_iter_sweep.py" \
        --index "$G/cagra/$ctag/index.cagra" \
        --base "$D/base.fbin" --query "$D/query.fbin" --gt "$gt" \
        --out "$out" --batch-size "$bs" --it-file "$ITF" \
        --k 10 --itopk-size 64 --search-width 1 --gpu 0 \
        --graph-degree "$deg" --intermediate-graph-degree $((deg * 2)) --nn-descent-niter 20 \
        --cache-dir "$G/cagra/$ctag/_search_cache" \
        > >(tee "$out/run.log") 2>&1
      ;;
    ggnn)
      "$BIN_DIR/ggnn_resident_search" \
        --base "$D/base.fvecs" --graph-dir "$G/ggnn/$gtag" \
        --query "$D/query.fbin" --gt "$gt" --out "$out" \
        --k-build "$deg" --k-query 10 --tau-query 0.5 \
        --batch-size "$bs" --gpu 0 --it-file "$ITF" \
        > >(tee "$out/run.log") 2>&1
      ;;
    ganns)
      "$BIN_DIR/ganns_resident_search_d$dim" \
        --base "$D/base.fvecs" --graph "$G/ganns/$atag/graph.nsw" \
        --query "$D/query.fbin" --gt "$gt" --out "$out" \
        --k 10 --M 512 --n-e 15 --batch-size "$bs" --gpu 0 --it-file "$ITF" \
        > >(tee "$out/run.log") 2>&1
      ;;
    song)
      "$BIN_DIR/song_resident_search_d${dim}_fd${deg}" \
        --base "$D/base.fbin" --graph "$G/song/$stag/bfsg.graph" \
        --query "$D/query.fbin" --gt "$gt" --out "$out" \
        --batch-size "$bs" --gpu 0 --n-base "$n" --dim "$dim" --shift "$shift" \
        --max-iterations 500 --kernel-chunk 0 --it-file "$ITS" \
        > >(tee "$out/run.log") 2>&1
      ;;
  esac
  rc=$?
  set -e
  {
    echo "dataset	sift1M"
    echo "algo	$algo"
    echo "batch_size	$bs"
    echo "gt	$gt"
    echo "exit_code	$rc"
    echo "finished_at	$(date '+%F %T')"
  } >>"$out/params.txt"
  if [[ $rc -ne 0 ]]; then
    log "FAIL $algo bs=$bs rc=$rc"
    fail=1
    return 0
  fi
  nrows=$(awk 'NR>1{c++} END{print c+0}' "$out/sweep.tsv")
  log "DONE $algo bs=$bs rows=$nrows last=[$(tail -1 "$out/sweep.tsv")]"
}

log "sift1M batch sweep start batches=$BATCHES"
for bs in $BATCHES; do
  run_one cagra "$bs"
  run_one ggnn "$bs"
  run_one ganns "$bs"
  run_one song "$bs"
done
log "ALL_DONE fail=$fail"
exit $fail
