#!/bin/bash
# FP32 and FP16 QPS-recall curves of Figure 14: GIST and DEEP1M; SONG, GGNN, GANNS, CAGRA; 16 curves.
# The iteration counts are 10, 12, ..., 500. At each iteration count the program runs one untimed warm-up search and then
# three timed searches; QPS = nq / kernel_ms of a timed search, as in Figure 7, and mean.tsv holds the mean over the
# timed searches.
# IT_SLEEP_MS (default 500) makes the programs sleep before each iteration count so that every measurement starts
# from an idle GPU. REPS (default 1) sweeps per curve; with more than one, the sweeps form the outer loop and
# mean.tsv pools the timed searches of all sweeps.
# usage: bash qps_curve.sh
# output: $OUT/<dataset>_<system>_<precision>/rep<i>/ (sweep.tsv, it_batches.tsv, logs) and mean.tsv
#   (OUT default $OUT_DIR/profiling/fp16_curves)
#   mean.tsv: it nq recall@10 reps qps_mean qps_std kernel_ms_mean per iteration count, over the timed searches
#   (reps = number of timed searches)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
OUT=${OUT:-$OUT_DIR/profiling/fp16_curves}
REPS=${REPS:-1}
export IT_SLEEP_MS=${IT_SLEEP_MS:-500}
GRID="$(seq 10 2 500)"
for i in $(seq 1 "$REPS"); do
  for ds in gist deep1M; do for sys in song ggnn ganns cagra; do for prec in fp32 fp16; do
    d=$OUT/${ds}_${sys}_${prec}/rep$i
    [[ -f $d/sweep.tsv ]] && continue
    mkdir -p "$d"
    L=$GRID
    echo "== rep$i $ds $sys $prec start $(date -Is)"
    IT_LIST="$L" bash "$HERE/run_point.sh" "$ds" "$sys" "$d" "$prec" > "$d/run.log" 2>&1 || echo "FAIL rep$i $ds $sys $prec"
    echo "== rep$i $ds $sys $prec rows=$(($(wc -l < "$d/sweep.tsv") - 1)) $(date -Is)"
  done; done; done
done

# Mean over the timed searches of each curve (it_batches.tsv rows timed1, timed2, ...), per iteration count
for g in "$OUT"/*_*_fp*/; do
  awk -F'\t' '
    FNR == 1 { for (c = 1; c <= NF; c++) col[$c] = c; next }
    $(col["phase"]) !~ /^timed/ { next }
    { it = $(col["it"]); q = $(col["nq"]) / $(col["kernel_ms"]) * 1000
      n[it]++; s[it] += q; ss[it] += q * q; k[it] += $(col["kernel_ms"]); rec[it] = $(col["recall@10"]); nq[it] = $(col["nq"])
      if (!(it in seen)) { seen[it] = 1; order[++m] = it } }
    END {
      print "it\tnq\trecall@10\treps\tqps_mean\tqps_std\tkernel_ms_mean"
      for (j = 1; j <= m; j++) { it = order[j]; mu = s[it] / n[it]
        v = (n[it] > 1) ? (ss[it] - n[it] * mu * mu) / (n[it] - 1) : 0; sd = (v > 0) ? sqrt(v) : 0
        printf "%s\t%s\t%s\t%d\t%.0f\t%.0f\t%.4f\n", it, nq[it], rec[it], n[it], mu, sd, k[it] / n[it] }
    }' "$g"rep*/it_batches.tsv > "$g/mean.tsv"
done
echo "curves aggregated: $(ls "$OUT"/*/mean.tsv | wc -l)"
