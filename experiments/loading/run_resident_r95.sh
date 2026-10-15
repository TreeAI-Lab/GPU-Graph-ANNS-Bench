#!/bin/bash
# Figure 6, deep1M and deep10M-annb: CAGRA, GGNN, GANNS and SONG with a GPU-resident index serve the
# 5,000,000 queries in batches of 10,000, at the smallest search effort that reaches 95% Recall@10.
#
#   1) calibration: on the first batch only (batch_id 0), find the minimal effort with
#      recall@10 >= TARGET; effort = max_iterations (CAGRA, GGNN, SONG) or n_e (GANNS).
#      Candidates CANDS (SONG needs at least 11); between the last failing and the first passing
#      candidate a binary search finds the minimum. If no candidate up to IT_MAX passes, IT_MAX is used
#      and its recall is recorded.
#   2) full run: all batches with the calibrated effort; per-batch H2D / kernel / D2H / recall.
#   3) summary: means over batches 1..end (batch 0 is the warm-up).
#
# Usage: bash experiments/loading/run_resident_r95.sh <deep1M|deep10M-annb>
#   ONLY=cagra|ggnn|ganns|song   one system
#   STEPS="calib full"           or only one of the two steps
#   MAX_BATCHES=N                limit the full run to N batches (0 = all)
#
# Outputs under $OUT_DIR/loading/<dataset>/:
#   <system>/calib_r95_b10000_k10/{calib_sweep.tsv,calib_chosen.txt,probe_it*/}
#   <system>/search_r1w1_b10000_it<IT>_k10_r95/{params.txt,load_timing.tsv,batch_timing.tsv,recall.tsv,run.log}
#   summary_resident_search_r95.tsv
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

HERE=$(cd "$(dirname "$0")" && pwd)
DS=${1:?usage: run_resident_r95.sh <deep1M|deep10M-annb>}
QUERY=${QUERY:-$DATA_DIR/deep100M/query.5M.fbin}
GT=${GT:-$DATA_DIR/$DS/gt.5M.k10.ibin}
BATCH=${BATCH:-10000}
K=${K:-10}
TARGET=${TARGET:-0.95}
IT_MAX=${IT_MAX:-256}
ONLY=${ONLY:-}
STEPS=${STEPS:-calib full}
MAX_BATCHES=${MAX_BATCHES:-0}
OUT=${OUT:-$OUT_DIR/loading/$DS}

# Candidate efforts (SONG needs >= 11)
CANDS=(11 16 20 24 32 40 48 55 64 80 96 128 160 200 256)

# Graphs: final degree about 32 for every system
CAGRA_TAG=${CAGRA_TAG:-gd32_igd64_it20}
GGNN_TAG=${GGNN_TAG:-k32_tau0.5_r2}
GANNS_TAG=${GANNS_TAG:-k16_m64}
SONG_TAG=${SONG_TAG:-fd32_shift6_deg16}
SONG_SHIFT=6
DIM=96

BASE_FBIN=$DATA_DIR/$DS/base.fbin
BASE_FVECS=$DATA_DIR/$DS/base.fvecs
CAGRA_IDX=$INDEX_DIR/$DS/cagra/$CAGRA_TAG/index.cagra
# A graph-only CAGRA index is rebuilt once with the dataset and cached here (see cagra_resident_search.py); the
# Figure 7 CAGRA sweep uses the same cache, so both search the same index
CAGRA_CACHE=$INDEX_DIR/$DS/cagra/$CAGRA_TAG/_search_cache
GGNN_DIR=$INDEX_DIR/$DS/ggnn/$GGNN_TAG
GANNS_GRAPH=$INDEX_DIR/$DS/ganns/$GANNS_TAG/graph.nsw
SONG_GRAPH=$INDEX_DIR/$DS/song/$SONG_TAG/bfsg.graph

GGNN_BIN=$BIN_DIR/ggnn_resident_search
GANNS_BIN=$BIN_DIR/ganns_resident_search_d$DIM
SONG_BIN=$BIN_DIR/song_resident_search_d${DIM}_fd32

[ -f "$GT" ] || { echo "missing GT $GT (run gt_5m.sh $DS)"; exit 2; }
[ -f "$QUERY" ] || { echo "missing QUERY $QUERY"; exit 2; }
[ -f "$BASE_FBIN" ] || { echo "missing $BASE_FBIN"; exit 2; }
N_BASE=$(od -An -t d4 -N 4 "$BASE_FBIN" | tr -d ' ')

maybe() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

ge_target() { awk -v a="$1" -v b="$TARGET" 'BEGIN { exit !(a + 0 >= b + 0) }'; }

# recall@10 of batch 0 from recall.tsv
read_recall0() {
  awk -F'\t' 'NR>1 && $1==0 {print $2; exit}' "$1"
}

# run_search <algo> <it> <out_dir> <max_batches, 0 = all>
run_search() {
  local algo=$1 it=$2 out=$3 maxb=$4
  local mb=()
  if [[ "$maxb" != "0" ]]; then mb=(--max-batches "$maxb"); fi
  mkdir -p "$out"
  case "$algo" in
    cagra)
      "$ENV_CAGRA/bin/python" "$HERE/cagra_resident_search.py" \
        --index "$CAGRA_IDX" --base "$BASE_FBIN" --query "$QUERY" --gt "$GT" \
        --out "$out" --batch-size "$BATCH" --k "$K" \
        --max-iterations "$it" --itopk-size 64 --search-width 1 \
        --gpu 0 --cache-dir "$CAGRA_CACHE" "${mb[@]+"${mb[@]}"}"
      ;;
    ggnn)
      "$GGNN_BIN" \
        --base "$BASE_FVECS" --graph-dir "$GGNN_DIR" --query "$QUERY" --gt "$GT" \
        --out "$out" --k-build 32 --k-query "$K" --tau-query 0.5 \
        --max-iterations "$it" --batch-size "$BATCH" --gpu 0 "${mb[@]+"${mb[@]}"}"
      ;;
    ganns)
      # Search M is the candidate-list size (a power of 2, >= n_e), independent of the build M.
      local gm=64
      while (( gm < it )); do gm=$((gm * 2)); done
      "$GANNS_BIN" \
        --base "$BASE_FVECS" --graph "$GANNS_GRAPH" --query "$QUERY" --gt "$GT" \
        --out "$out" --k "$K" --M "$gm" --n-e "$it" \
        --batch-size "$BATCH" --gpu 0 "${mb[@]+"${mb[@]}"}"
      ;;
    song)
      "$SONG_BIN" \
        --base "$BASE_FBIN" --graph "$SONG_GRAPH" --query "$QUERY" --gt "$GT" \
        --out "$out" --batch-size "$BATCH" --gpu 0 \
        --n-base "$N_BASE" --dim "$DIM" --shift "$SONG_SHIFT" \
        --max-iterations "$it" --kernel-chunk 0 "${mb[@]+"${mb[@]}"}"
      ;;
  esac
}

run_probe() {
  local algo=$1 it=$2 out=$3
  rm -rf "$out"
  mkdir -p "$out"
  run_search "$algo" "$it" "$out" 1 >"$out/run.log" 2>&1
}

calib_one() {
  local algo=$1
  local caldir=$OUT/$algo/calib_r95_b${BATCH}_k${K}
  mkdir -p "$caldir"
  local sweep=$caldir/calib_sweep.tsv
  echo -e "it\trecall@10\tprobe_dir" >"$sweep"
  echo "===== CALIB $DS $algo target=$TARGET ====="

  local chosen="" chosen_r=""
  local prev_it="" prev_r=""
  declare -A SEEN=()

  for it in "${CANDS[@]}"; do
    (( it > IT_MAX )) && continue
    [[ -n "${SEEN[$it]:-}" ]] && continue
    SEEN[$it]=1
    local pout=$caldir/probe_it${it}
    echo "[calib] $algo probe it=$it ..."
    run_probe "$algo" "$it" "$pout"
    local r
    r=$(read_recall0 "$pout/recall.tsv" || true)
    if [[ -z "$r" ]]; then
      echo "FAILED: no recall for $algo it=$it"; tail -30 "$pout/run.log"; exit 2
    fi
    echo -e "${it}\t${r}\t${pout}" >>"$sweep"
    echo "[calib] $algo it=$it recall=$r"
    if ge_target "$r"; then
      # binary search between prev_it + 1 and it when the previous candidate failed
      if [[ -n "$prev_it" ]] && ! ge_target "$prev_r"; then
        local lo=$((prev_it + 1)) hi=$it
        while (( lo < hi )); do
          local mid=$(( (lo + hi) / 2 ))
          [[ -n "${SEEN[$mid]:-}" ]] && { lo=$((mid + 1)); continue; }
          SEEN[$mid]=1
          local mout=$caldir/probe_it${mid}
          echo "[calib] $algo refine mid=$mid lo=$lo hi=$hi"
          run_probe "$algo" "$mid" "$mout"
          local mr
          mr=$(read_recall0 "$mout/recall.tsv")
          echo -e "${mid}\t${mr}\t${mout}" >>"$sweep"
          echo "[calib] $algo it=$mid recall=$mr"
          if ge_target "$mr"; then
            hi=$mid
            chosen=$mid; chosen_r=$mr
          else
            lo=$((mid + 1))
            prev_it=$mid; prev_r=$mr
          fi
        done
        # hi is the smallest passing effort
        if [[ -z "$chosen" ]] || (( hi < chosen )); then
          chosen=$hi
          if [[ ! -f $caldir/probe_it${hi}/recall.tsv ]]; then
            run_probe "$algo" "$hi" "$caldir/probe_it${hi}"
            chosen_r=$(read_recall0 "$caldir/probe_it${hi}/recall.tsv")
            echo -e "${hi}\t${chosen_r}\t$caldir/probe_it${hi}" >>"$sweep"
          else
            chosen_r=$(read_recall0 "$caldir/probe_it${hi}/recall.tsv")
          fi
        fi
      else
        chosen=$it; chosen_r=$r
      fi
      break
    fi
    prev_it=$it; prev_r=$r
  done

  if [[ -z "$chosen" ]]; then
    chosen=$IT_MAX
    local pout=$caldir/probe_it${IT_MAX}
    if [[ ! -f $pout/recall.tsv ]]; then
      run_probe "$algo" "$IT_MAX" "$pout"
    fi
    chosen_r=$(read_recall0 "$pout/recall.tsv")
    echo -e "${IT_MAX}\t${chosen_r}\t${pout}" >>"$sweep"
    echo "[calib] WARNING $algo never reached $TARGET; using IT_MAX=$IT_MAX recall=$chosen_r"
  fi

  {
    echo "dataset	$DS"
    echo "algo	$algo"
    echo "target_recall@10	$TARGET"
    echo "chosen_it	$chosen"
    echo "achieved_recall@10	$chosen_r"
    echo "batch_size	$BATCH"
    echo "k	$K"
    echo "query	$QUERY"
    echo "gt	$GT"
    echo "calib_dir	$caldir"
  } >"$caldir/calib_chosen.txt"
  echo "[calib] CHOSEN $DS $algo it=$chosen recall=$chosen_r -> $caldir/calib_chosen.txt"
}

run_full() {
  local algo=$1
  local cal=$OUT/$algo/calib_r95_b${BATCH}_k${K}/calib_chosen.txt
  [ -f "$cal" ] || { echo "missing $cal: run the calibration first"; exit 2; }
  local it
  it=$(awk -F'\t' '$1=="chosen_it"{print $2; exit}' "$cal")
  local tag=search_r1w1_b${BATCH}_it${it}_k${K}_r95
  local out=$OUT/$algo/$tag
  mkdir -p "$out"
  echo "======== FULL $DS $algo it=$it -> $out ========"
  run_search "$algo" "$it" "$out" "$MAX_BATCHES" 2>&1 | tee "$out/run.log"
  {
    echo "protocol	r95_first_batch_calib"
    echo "calib_file	$cal"
    cat "$cal"
  } >>"$out/params.txt"
}

mkdir -p "$OUT"
exec > >(tee -a "$OUT/run_resident_r95.log") 2>&1
echo "===== start $DS steps=($STEPS) $(date '+%F %T') ====="

for step in $STEPS; do
  for algo in cagra ggnn ganns song; do
    maybe "$algo" || continue
    case "$step" in
      calib) calib_one "$algo" ;;
      full) run_full "$algo" ;;
      *) echo "bad step $step"; exit 2 ;;
    esac
  done
done

"$ENV_BUILD/bin/python" "$HERE/resident_summary.py" --root "$OUT" --batch "$BATCH" --k "$K" \
  --systems cagra ggnn ganns song
echo "===== done $DS $(date '+%F %T') ====="
