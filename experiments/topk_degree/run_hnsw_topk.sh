#!/bin/bash
# Figure 11, HNSW curves: Faiss HNSW (IndexHNSWFlat, M = 32, efConstruction = 200, L2) on sift1M with
# 1, 10, 20 and 40 threads for K = 10, 40, 70, 100.
# For every K the efSearch values of ef_schedule.txt that are >= K are searched (plus K itself);
# all 10,000 queries go to one search() call; each efSearch is searched once untimed as warm-up,
# then timed SEARCH_REPEATS (3) times; wall_ms is the mean of the timed searches, QPS = nq / mean
# wall time; recall = Recall@K.
#
# Usage: bash experiments/topk_degree/run_hnsw_topk.sh
# Outputs under $OUT_DIR/topk_degree/hnsw/:
#   ef_schedule_k<K>.txt, k<K>/t<T>/{params.txt,run.log,sweep.tsv}
#   summary.tsv   dataset algo degree efSearch k threads nq wall_ms qps recall wall_ms_std n_timed
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$BIN_DIR/faiss_hnsw}
DATA=$DATA_DIR/sift1M
OUT=${OUT:-$OUT_DIR/topk_degree/hnsw}
INDEX=${INDEX:-$INDEX_DIR/sift1M/hnsw/M32_efc200/hnsw_M32.faissindex}
EF_ALL=${EF_ALL:-$HERE/ef_schedule.txt}
KS=${KS:-10 40 70 100}
THREADS=${THREADS:-1 10 20 40}
BUILD_THREADS=${BUILD_THREADS:-10}
DEGREE=32
EFC=200
export SEARCH_REPEATS=${SEARCH_REPEATS:-3}
export LD_LIBRARY_PATH=$W/build/faiss/install/lib:$ENV_FAISS/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$OUT" "$(dirname "$INDEX")"

if [[ ! -f "$INDEX" ]]; then
  echo "=== build IndexHNSWFlat M=$DEGREE efConstruction=$EFC threads=$BUILD_THREADS -> $INDEX ==="
  "$BIN" build "$DATA" "$DEGREE" "$EFC" "$BUILD_THREADS" "$INDEX" 2>&1 | tee "$OUT/build.log"
fi

SUMMARY=$OUT/summary.tsv
echo -e "dataset\talgo\tdegree\tefSearch\tk\tthreads\tnq\twall_ms\tqps\trecall\twall_ms_std\tn_timed" > "$SUMMARY"

# efSearch values >= k, with k itself added if missing
filter_ef() {
  local k=$1
  local out=$2
  awk -v k="$k" '$1+0 >= k {print $1+0}' "$EF_ALL" > "$out"
  if ! awk -v k="$k" '$1+0==k{found=1} END{exit !found}' "$out"; then
    { echo "$k"; cat "$out"; } | sort -n -u > "${out}.tmp"
    mv "${out}.tmp" "$out"
  fi
}

for K in $KS; do
  EF_K=$OUT/ef_schedule_k${K}.txt
  filter_ef "$K" "$EF_K"
  N_EF=$(wc -l < "$EF_K")
  for T in $THREADS; do
    DIR=$OUT/k${K}/t${T}
    mkdir -p "$DIR"
    cat > "$DIR/params.txt" <<EOF
algo	faiss_hnsw
dataset	sift1M
degree	${DEGREE}
efConstruction	${EFC}
ef_file	${EF_K}
n_ef	${N_EF}
k	${K}
threads	${T}
batch_size	all
search_repeats	${SEARCH_REPEATS}
index	${INDEX}
protocol	full_nq_one_search; warmup1_timed${SEARCH_REPEATS}_mean; efSearch>=k ladder
EOF
    echo "=== k=$K threads=$T n_ef=$N_EF ==="
    "$BIN" search "$DATA" "$INDEX" "$K" "$K" \
      --threads "$T" --ef-file "$EF_K" \
      2>&1 | tee "$DIR/run.log"

    {
      echo -e "threads\tbatch\tefSearch\tk\tnq\tn_batches\twall_ms\tqps\trecall@${K}\twall_ms_std\tn_timed"
      grep '^row' "$DIR/run.log" | cut -f2-
    } > "$DIR/sweep.tsv"

    while IFS=$'\t' read -r threads batch ef k nq n_batches wall_ms qps recall wall_ms_std n_timed; do
      [[ "$threads" == "threads" ]] && continue
      printf '%s\n' \
        "sift1M	faiss_hnsw	${DEGREE}	${ef}	${k}	${threads}	${nq}	${wall_ms}	${qps}	${recall}	${wall_ms_std}	${n_timed}" \
        >> "$SUMMARY"
    done < "$DIR/sweep.tsv"
  done
done

echo "wrote $SUMMARY ($(grep -c $'^sift1M' "$SUMMARY" || true) rows)"
