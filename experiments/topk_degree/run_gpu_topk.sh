#!/bin/bash
# Figure 11, GPU and hybrid systems: CAGRA, GGNN, GANNS, SONG, BANG and PilotANN on SIFT1M for K = 40, 70, 100.
# Each system runs its Figure 7 search program on its Figure 7 index (experiments/qps_recall) with K neighbors
# returned, and recall is Recall@K against the top-K ground truth. The K = 10 curves of Figure 11 are those of
# Figure 7. Every point searches the 10,000 queries as one batch, once as an untimed warm-up and then three times
# timed; sweep.tsv holds the means.
# Sweeps: the Figure 7 schedules without the points that cannot return K neighbors:
#   CAGRA, GGNN  every count of it_schedule.txt (CAGRA's internal top-k is at least K, see cagra_iter_sweep.py)
#   GANNS        the n_e of it_schedule.txt whose candidate list (64, doubled until it holds n_e) holds K
#   SONG         the counts of it_schedule_song.txt above K (SONG returns max_iterations - 1 neighbors)
#   BANG         L = K, K + BANG_L_STEP, ... up to 512 (automatic mode of bang_search)
#   PilotANN     the ef values of Figure 7 that are >= K
#
# Usage:  bash experiments/topk_degree/run_gpu_topk.sh
#         KS="40 70 100"  ONLY=<cagra|ggnn|ganns|song|bang|pilotann>  FORCE=1 repeats finished sweeps
# Needs:  the SIFT1M indexes and search programs of experiments/qps_recall (build_graphs.sh sift1M).
# Output: $OUT_DIR/topk_degree/gpu/gt.k<K>.ibin
#         $OUT_DIR/topk_degree/gpu/k<K>/<system>/ (it.txt, sweep.tsv, params.txt, run.log; BANG raw.log)
#         then: python experiments/topk_degree/summarize_gpu_topk.py --root $OUT_DIR/topk_degree/gpu
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)
QR=$BENCH/experiments/qps_recall
source "$QR/datasets.sh"

KS=${KS:-40 70 100}
ONLY=${ONLY:-}
FORCE=${FORCE:-0}
DS=sift1M
OUT=$OUT_DIR/topk_degree/gpu
PY=${PY:-$ENV_BUILD/bin/python}
PY_CAGRA=$ENV_CAGRA/bin/python
BANG_DIR=${BANG_DIR:-$W/build/bang/r128}
export BANG_L_STEP=${BANG_L_STEP:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-14}
MAX_L=512                          # largest worklist length of bang_search

read -r data n dim deg nq ctag gtag atag stag shift <<<"$(cfg_graph "$DS")"
read -r R L_BUILD M_BANG B QD <<<"$(cfg_bang "$DS")"
read -r gtype nn efc dp ratio <<<"$(cfg_pilotann "$DS")"
D=$DATA_DIR/$data
G=$INDEX_DIR/$data
BANG_PREFIX=$G/bang/$(bang_tag "$R" "$L_BUILD")/index
PILOT_CKPT=$G/pilotann/$(pilotann_tag "$gtype" "$nn" "$efc" "$dp" "$ratio")/index.pt
mkdir -p "$OUT"

maybe() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

# schedule <system> <K>: the effort values of the sweep, one per line
schedule() {
  local algo=$1 k=$2
  case $algo in
    cagra|ggnn) grep -v '^#' "$QR/it_schedule.txt" | awk 'NF' ;;
    ganns) grep -v '^#' "$QR/it_schedule.txt" |
             awk -v k="$k" 'NF { m = 64; while (m < $1) m *= 2; if (m >= k) print $1 }' ;;
    song) grep -v '^#' "$QR/it_schedule_song.txt" | awk -v k="$k" 'NF && $1 > k' ;;
    bang) seq "$k" "$BANG_L_STEP" "$MAX_L" ;;
    # the ef values of Figure 7 (DEFAULT_EFS of src/search/pilotann/pilotann_sweep.py)
    pilotann) { seq 16 4 64; seq 68 8 128; seq 144 16 256; seq 288 32 512; } | awk -v k="$k" '$1 >= k' ;;
  esac
}

run_one() {
  local algo=$1 k=$2 out gt itf need nrows rc bang_rc=""
  out=$OUT/k$k/$algo
  gt=$OUT/gt.k$k.ibin
  [[ -f $gt ]] || "$PY" "$HERE/gt_topk.py" "$D/gt.ibin" "$gt" "$k"
  mkdir -p "$out"
  itf=$out/it.txt
  schedule "$algo" "$k" >"$itf"
  need=$(awk 'END { print NR }' "$itf")
  if [[ -f $out/sweep.tsv && $FORCE != 1 ]]; then
    nrows=$(awk 'NR > 1 { c++ } END { print c + 0 }' "$out/sweep.tsv")
    if [[ $nrows -ge $need ]]; then
      echo "SKIP complete K=$k $algo ($nrows rows)"
      return 0
    fi
  fi
  if [[ $algo == bang && ! -s ${BANG_PREFIX}_disk.bin ]]; then
    echo "missing BANG index ${BANG_PREFIX}_disk.bin" >&2
    return 2
  fi
  echo "======== K=$k $algo ($need points) ========"
  set +e
  case $algo in
    cagra)
      # The index is graph-only: the first run rebuilds it with the dataset and caches it (as in Figure 7)
      "$PY_CAGRA" "$BENCH/src/search/cagra/cagra_iter_sweep.py" \
        --index "$G/cagra/$ctag/index.cagra" \
        --base "$D/base.fbin" --query "$D/query.fbin" --gt "$gt" \
        --out "$out" --batch-size "$nq" --it-file "$itf" \
        --k "$k" --itopk-size 64 --search-width 1 --gpu 0 \
        --graph-degree "$deg" --intermediate-graph-degree "$((deg * 2))" --nn-descent-niter 20 \
        --cache-dir "$G/cagra/$ctag/_search_cache" \
        > >(tee "$out/run.log") 2>&1
      rc=$?
      ;;
    ggnn)
      "$BIN_DIR/ggnn_resident_search" \
        --base "$D/base.fvecs" --graph-dir "$G/ggnn/$gtag" \
        --query "$D/query.fbin" --gt "$gt" --out "$out" \
        --k-build "$deg" --k-query "$k" --tau-query 0.5 \
        --batch-size "$nq" --gpu 0 --it-file "$itf" \
        > >(tee "$out/run.log") 2>&1
      rc=$?
      ;;
    ganns)
      "$BIN_DIR/ganns_resident_search_d$dim" \
        --base "$D/base.fvecs" --graph "$G/ganns/$atag/graph.nsw" \
        --query "$D/query.fbin" --gt "$gt" --out "$out" \
        --k "$k" --M 512 --n-e 15 --batch-size "$nq" --gpu 0 --it-file "$itf" \
        > >(tee "$out/run.log") 2>&1
      rc=$?
      ;;
    song)
      "$BIN_DIR/song_resident_search_d${dim}_fd${deg}" \
        --base "$D/base.fbin" --graph "$G/song/$stag/bfsg.graph" \
        --query "$D/query.fbin" --gt "$gt" --out "$out" \
        --batch-size "$nq" --gpu 0 --n-base "$n" --dim "$dim" --shift "$shift" \
        --max-iterations 500 --kernel-chunk 0 --it-file "$itf" --recall-k "$k" \
        > >(tee "$out/run.log") 2>&1
      rc=$?
      ;;
    bang)
      # 7 arguments: automatic L sweep from L = K
      LD_LIBRARY_PATH=$BANG_DIR:$ENV_BUILD/lib:${LD_LIBRARY_PATH:-} \
        "$BANG_DIR/bang_search" "$BANG_PREFIX" "$D/query.fbin" "$gt" "$nq" "$k" float l2 >"$out/raw.log" 2>&1
      bang_rc=$?
      "$PY" "$QR/parse_bang_raw.py" --raw "$out/raw.log" --out "$out" --nq "$nq"
      rc=$?
      ;;
    pilotann)
      LD_LIBRARY_PATH=$ENV_PILOT/lib:${LD_LIBRARY_PATH:-} \
        "$ENV_PILOT/bin/python" -u "$BENCH/src/search/pilotann/pilotann_sweep.py" --dataset "$DS" \
        --ckpt "$PILOT_CKPT" --query "$D/query.fbin" --gt "$gt" --out "$out" --batch "$nq" \
        --k "$k" --efs "$(paste -sd, "$itf")" \
        > >(tee "$out/run.log") 2>&1
      rc=$?
      ;;
  esac
  set -e
  {
    echo "dataset	$DS"
    echo "algo	$algo"
    echo "k	$k"
    echo "batch_size	$nq"
    echo "max_degree	$deg"
    echo "schedule	$itf"
    echo "gt	$gt"
    [[ -z $bang_rc ]] || echo "bang_exit	$bang_rc"
    echo "exit_code	$rc"
  } >>"$out/params.txt"
  if [[ $rc -ne 0 ]]; then
    echo "FAIL K=$k $algo rc=$rc" >&2
    return "$rc"
  fi
}

fail=0
for k in $KS; do
  for algo in cagra ggnn ganns song bang pilotann; do
    maybe "$algo" || continue
    run_one "$algo" "$k" || fail=1
  done
done
echo "GPU top-K runs in $OUT (fail=$fail)"
exit $fail
