#!/bin/bash
# One search at the operating point of points.tsv, in the sweep mode of each program with an iteration file that
# holds only the operating point. SONG, GGNN, GANNS and CAGRA run one untimed warm-up search and then three timed
# searches, so their main kernel is launched exactly four times (Nsight Compute skips the first launch and measures
# the second, the first timed search). BANG runs four searches at one L (one warm-up, three timed) and launches its
# main kernel once per iteration. The search parameters are those of the Figure 7 QPS-recall sweep.
#
# usage: run_point.sh <gist|deep1M> <song|ggnn|ganns|cagra|bang> <out dir> [fp32|fp16]
#   The FP32 and FP16 programs of SONG, GGNN, GANNS and CAGRA are built by src/fp16/build.sh.
#   IT_LIST="15 16 ..." searches these iteration counts instead of the operating point (used by qps_curve.sh).
#   PRE_SLEEP_S sleeps before the run, so that the GPU starts from idle and the previous load does not
#   affect this one when the clocks are not locked.
# output: <out dir>/sweep.tsv (it, nq, kernel_ms, recall@10, ... as written by each program; times are the means
#   over the timed searches) and, for SONG, GGNN, GANNS and CAGRA, it_batches.tsv (every search, phase warmup or
#   timed1 to timed3)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
[[ -n "${PRE_SLEEP_S:-}" ]] && sleep "$PRE_SLEEP_S"
DS=${1:?dataset}
SYS=${2:?system}
OUT=${3:?out dir}
PREC=${4:-fp32}
PY=${PY:-$ENV_BUILD/bin/python}
D=$DATA_DIR/$DS
GT=$OUT_DIR/profiling/gt/${DS}_k10.ibin   # top-10 ground truth; GANNS needs exactly k columns

case "$DS" in
  deep1M) N=1000000; DIM=96; DEG=32; SHIFT=6; NQ=10000; BANG_R=32
          CTAG=gd32_igd64_it20; GTAG=k32_tau0.5_r2; ATAG=k16_m64; STAG=fd32_shift6_deg16 ;;
  gist)   N=1000000; DIM=960; DEG=128; SHIFT=8; NQ=1000; BANG_R=128
          CTAG=gd128_igd256_it20; GTAG=k128_tau0.5_r2; ATAG=k64_m256; STAG=fd128_shift8_deg64 ;;
  *) echo "unknown dataset $DS" >&2; exit 1 ;;
esac
# Graphs of Figure 7 (experiments/qps_recall/build_graphs.sh); CAGRA uses the index with its dataset attached that
# the Figure 7 CAGRA sweep caches in _search_cache/ (src/search/cagra/cagra_py_common.py)
SONG_GRAPH=$INDEX_DIR/$DS/song/$STAG/bfsg.graph
GGNN_GRAPH=$INDEX_DIR/$DS/ggnn/$GTAG
GANNS_GRAPH=$INDEX_DIR/$DS/ganns/$ATAG/graph.nsw
CAGRA_INDEX=$INDEX_DIR/$DS/cagra/$CTAG/_search_cache/index_with_dataset.cagra
BANG_PREFIX=$INDEX_DIR/$DS/bang/r${BANG_R}_l100_b4/bang_format/$DS   # build_bang_graphs.sh
BANG_BIN=$W/build/bang/r$BANG_R/bang_search                          # third_party/build/bang.sh <R>

IT=$(awk -F'\t' -v d="$DS" -v s="$SYS" '$1==d && $2==s {print $3}' "$HERE/points.tsv")
[[ "$IT" =~ ^[0-9]+$ ]] || { echo "no operating point for $DS $SYS in points.tsv" >&2; exit 2; }
if [[ ! -f "$GT" ]]; then
  mkdir -p "$(dirname "$GT")"
  "$PY" "$HERE/make_gt_k10.py" "$D/gt.ibin" "$GT"
fi
mkdir -p "$OUT"
if [[ -n "${IT_LIST:-}" ]]; then tr ' ' '\n' <<< "$IT_LIST" | grep . > "$OUT/it_point.txt"; else echo "$IT" > "$OUT/it_point.txt"; fi
echo "[run_point] $DS $SYS $PREC it=$IT out=$OUT"

case "$SYS" in
  cagra)
    # Same float index and graph for both precisions; fp16 converts the dataset and the queries to half
    "$BIN_DIR/cagra_fp16" --index "$CAGRA_INDEX" --query "$D/query.fbin" --gt "$GT" \
      --out "$OUT" --batch-size 10000 --it-file "$OUT/it_point.txt" --k 10 --precision "$PREC" ;;
  ggnn)
    "$BIN_DIR/ggnn_${DS}_$PREC" --base "$D/base.fvecs" --graph-dir "$GGNN_GRAPH" \
      --query "$D/query.fbin" --gt "$GT" --out "$OUT" --k-build "$DEG" --k-query 10 --tau-query 0.5 \
      --batch-size 10000 --gpu 0 --it-file "$OUT/it_point.txt" ;;
  ganns)
    "$BIN_DIR/ganns_${DS}_$PREC" --base "$D/base.fvecs" --graph "$GANNS_GRAPH" \
      --query "$D/query.fbin" --gt "$GT" --out "$OUT" --k 10 --M 512 --n-e 15 --batch-size 10000 --gpu 0 \
      --it-file "$OUT/it_point.txt" ;;
  song)
    "$BIN_DIR/song_${DS}_$PREC" --base "$D/base.fbin" --graph "$SONG_GRAPH" \
      --query "$D/query.fbin" --gt "$GT" --out "$OUT" --batch-size 10000 --gpu 0 --n-base "$N" --dim "$DIM" \
      --shift "$SHIFT" --max-iterations 500 --kernel-chunk 0 --it-file "$OUT/it_point.txt" ;;
  bang)
    # FP32 only. An integer 8th argument runs only this L: one warm-up search, then three timed searches.
    OMP_NUM_THREADS=${OMP_NUM_THREADS:-14} "$BANG_BIN" "$BANG_PREFIX" \
      "$D/query.fbin" "$GT" "$NQ" 10 float l2 "$IT" > "$OUT/bang.log" 2>&1
    # After the timed searches of an L, BANG prints one summary line "L ms QPS recall(%) [ms_std n_timed]" with the
    # mean wall time of the timed searches; keep the last one
    awk -v nq="$NQ" 'BEGIN { print "it\tnq\twall_ms\tqps\trecall@10\twall_ms_std\tn_timed" }
      NF >= 4 && $1 ~ /^[0-9]+$/ && $4 ~ /^[0-9.]+$/ {
        last = sprintf("%s\t%s\t%s\t%s\t%.6f\t%s\t%s", $1, nq, $2, $3, $4 / 100, (NF >= 6 ? $5 : ""), (NF >= 6 ? $6 : "")) }
      END { print last }' "$OUT/bang.log" > "$OUT/sweep.tsv" ;;
  *) echo "unknown system $SYS" >&2; exit 1 ;;
esac
echo "[run_point] done; recall in $OUT/sweep.tsv"
