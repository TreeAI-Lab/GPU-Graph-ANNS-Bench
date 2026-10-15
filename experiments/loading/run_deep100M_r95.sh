#!/bin/bash
# Figure 6, deep100M: CAGRA with a GPU-resident index serves the 5,000,000 queries in batches of 10,000
# at the smallest max_iterations that reaches 95% Recall@10 on the first batch.
#
# The graph-only index (built with the IVF-PQ graph builder) gets the base vectors attached in-process.
# itopk_size follows max_iterations: max(32, k, it) rounded up to a multiple of 32.
# Steps: ground truth of the 5M queries (gt_5m.sh, if missing), calibration on batch 0 over CANDS with
# a binary search between the last failing and the first passing candidate, full run, summary
# (means over batches 1..end).
#
# Usage: bash experiments/loading/run_deep100M_r95.sh
# Outputs under $OUT_DIR/loading/deep100M/cagra/:
#   calib_r95_b10000_k10_itopklin/{calib_sweep.tsv,calib_chosen.txt,probe_it*/}
#   search_r1w1_b10000_it<IT>_k10_r95/{params.txt,load_timing.tsv,batch_timing.tsv,recall.tsv,run.log}
# and $OUT_DIR/loading/deep100M/summary_resident_search_r95.tsv
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

HERE=$(cd "$(dirname "$0")" && pwd)
DS=deep100M
OUT=${OUT:-$OUT_DIR/loading/$DS}
BIN=${BIN:-$BIN_DIR/cagra_resident_search}
CAGRA_TAG=${CAGRA_TAG:-gd32_igd64_ivfpq}
INDEX=$INDEX_DIR/$DS/cagra/$CAGRA_TAG/index.cagra
BASE=$DATA_DIR/$DS/base.fbin
QUERY=${QUERY:-$DATA_DIR/$DS/query.5M.fbin}
GT=${GT:-$DATA_DIR/$DS/gt.5M.k10.ibin}
BATCH=${BATCH:-10000}
K=${K:-10}
TARGET=${TARGET:-0.95}
IT_MAX=${IT_MAX:-512}
CANDS=(11 16 20 24 32 40 48 55 64 80 96 128 160 200 256 320 384 448 512)
CALDIR=$OUT/cagra/calib_r95_b${BATCH}_k${K}_itopklin

[ -x "$BIN" ] || { echo "missing $BIN"; exit 2; }
[ -f "$INDEX" ] || { echo "missing $INDEX"; exit 2; }
[ -f "$QUERY" ] || { echo "missing QUERY $QUERY"; exit 2; }
if [ ! -f "$GT" ]; then
  QUERY="$QUERY" GT="$GT" bash "$HERE/gt_5m.sh" "$DS"
fi

mkdir -p "$OUT/cagra"

read_recall0() {
  awk -F'\t' 'NR>1 && $1==0 {print $2; exit}' "$1"
}

itopk_for_it() {
  local it=$1 need=$1
  if (( need < K )); then need=$K; fi
  if (( need < 32 )); then need=32; fi
  echo $(( ((need + 31) / 32) * 32 ))
}

# run_search <it> <out_dir> <max_batches, 0 = all>
run_search() {
  local it=$1 out=$2 maxb=${3:-0}
  local itopk
  itopk=$(itopk_for_it "$it")
  mkdir -p "$out"
  echo "[cagra] it=$it itopk=$itopk max_batches=$maxb"
  "$BIN" \
    --index "$INDEX" --base "$BASE" --query "$QUERY" --gt "$GT" \
    --out "$out" --batch-size "$BATCH" --k "$K" \
    --max-iterations "$it" --itopk-size "$itopk" --search-width 1 \
    --max-batches "$maxb" --gpu 0 \
    2>&1 | tee "$out/run.log"
}

ge_target() { awk -v a="$1" -v b="$TARGET" 'BEGIN { exit !(a + 0 >= b + 0) }'; }

calib_one() {
  local caldir=$CALDIR
  mkdir -p "$caldir"
  local sweep=$caldir/calib_sweep.tsv
  echo -e "it\titopk\trecall@10\tprobe_dir" >"$sweep"
  echo "===== CALIB $DS cagra target=$TARGET itopk=max(32,ceil32(it)) ====="

  local chosen="" chosen_r="" prev_it="" prev_r=""
  declare -A SEEN=()

  for it in "${CANDS[@]}"; do
    (( it > IT_MAX )) && continue
    [[ -n "${SEEN[$it]:-}" ]] && continue
    SEEN[$it]=1
    local pout=$caldir/probe_it${it}
    echo "[calib] cagra probe it=$it ..."
    run_search "$it" "$pout" 1
    local r
    r=$(read_recall0 "$pout/recall.tsv" || true)
    if [[ -z "$r" ]]; then
      echo "FAILED: no recall for it=$it"; tail -40 "$pout/run.log"; exit 2
    fi
    echo -e "${it}\t$(itopk_for_it "$it")\t${r}\t${pout}" >>"$sweep"
    echo "[calib] cagra it=$it itopk=$(itopk_for_it "$it") recall=$r"
    if ge_target "$r"; then
      if [[ -n "$prev_it" ]] && ! ge_target "$prev_r"; then
        local lo=$((prev_it + 1)) hi=$it
        while (( lo < hi )); do
          local mid=$(( (lo + hi) / 2 ))
          [[ -n "${SEEN[$mid]:-}" ]] && { lo=$((mid + 1)); continue; }
          SEEN[$mid]=1
          local mout=$caldir/probe_it${mid}
          echo "[calib] cagra refine mid=$mid lo=$lo hi=$hi"
          run_search "$mid" "$mout" 1
          local mr
          mr=$(read_recall0 "$mout/recall.tsv")
          echo -e "${mid}\t$(itopk_for_it "$mid")\t${mr}\t${mout}" >>"$sweep"
          echo "[calib] cagra it=$mid itopk=$(itopk_for_it "$mid") recall=$mr"
          if ge_target "$mr"; then
            hi=$mid
            chosen=$mid; chosen_r=$mr
          else
            lo=$((mid + 1))
            prev_it=$mid; prev_r=$mr
          fi
        done
        if [[ -z "$chosen" ]] || (( hi < chosen )); then
          chosen=$hi
          chosen_r=$(read_recall0 "$caldir/probe_it${hi}/recall.tsv")
        fi
      else
        chosen=$it; chosen_r=$r
      fi
      break
    fi
    prev_it=$it; prev_r=$r
  done

  if [[ -z "$chosen" ]]; then
    echo "[calib] cagra never reached $TARGET by it=$IT_MAX" >&2
    exit 2
  fi

  {
    echo "dataset	$DS"
    echo "algo	cagra"
    echo "target_recall@10	$TARGET"
    echo "itopk_policy	max(32, ceil32(it))"
    echo "chosen_it	$chosen"
    echo "chosen_itopk	$(itopk_for_it "$chosen")"
    echo "achieved_recall@10	$chosen_r"
    echo "batch_size	$BATCH"
    echo "k	$K"
    echo "query	$QUERY"
    echo "gt	$GT"
    echo "calib_dir	$caldir"
  } >"$caldir/calib_chosen.txt"
  echo "[calib] CHOSEN $DS cagra it=$chosen itopk=$(itopk_for_it "$chosen") recall=$chosen_r"
}

run_full() {
  local it
  it=$(awk -F'\t' '$1=="chosen_it"{print $2; exit}' "$CALDIR/calib_chosen.txt")
  local tag=search_r1w1_b${BATCH}_it${it}_k${K}_r95
  local out=$OUT/cagra/$tag
  echo "======== FULL $DS cagra it=$it -> $out ========"
  run_search "$it" "$out" 0
  {
    echo "protocol	r95_first_batch_calib"
    echo "calib_file	$CALDIR/calib_chosen.txt"
    cat "$CALDIR/calib_chosen.txt"
  } >>"$out/params.txt"
}

exec > >(tee -a "$OUT/run_resident_r95.log") 2>&1
echo "===== start $DS cagra $(date '+%F %T') ====="
calib_one
run_full
"$ENV_BUILD/bin/python" "$HERE/resident_summary.py" --root "$OUT" --batch "$BATCH" --k "$K" \
  --systems cagra --calib-suffix _itopklin
echo "===== done $DS cagra $(date '+%F %T') ====="
