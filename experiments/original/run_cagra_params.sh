#!/bin/bash
# Figure 13: CAGRA on SIFT1M with one search parameter varied at a time (search width, internal top-k, team
# size) on the degree-32 graph, and the base setting on graphs of degree 16 to 128. Recall@10 of 10,000
# queries per setting. my_cagra prints no search time; the time column of Figure 13 has no code here.
# The base setting of the original sweep is not recorded; the defaults are those of run_cagra.sh.
#
# Usage:  bash run_cagra_params.sh
# Needs:  build_cagra.sh, build_indexes.sh cagra sift1M <degree> for every degree in DEGREES
# Output: $OUT_DIR/original/cagra/<dataset>-<parameter>-<value>/k10/it<it>.log, .topk, .recall
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

DS=${DS:-sift1M}
IT=${IT:-32}
BASE_SEARCH_WIDTH=${BASE_SEARCH_WIDTH:-1}
BASE_ITOPK=${BASE_ITOPK:-128}
BASE_TEAM_SIZE=${BASE_TEAM_SIZE:-8}
BASE_DEGREE=${BASE_DEGREE:-32}
SEARCH_WIDTHS=${SEARCH_WIDTHS:-"1 2 4 8 16 32"}
ITOPKS=${ITOPKS:-"16 32 64 128 256"}
TEAM_SIZES=${TEAM_SIZES:-"8 16 32"}
DEGREES=${DEGREES:-"16 32 64 96 128"}
K=10

search() {
  local tag=$1 deg=$2 sw=$3 itopk=$4 team=$5
  local out=$ORIG_OUT/cagra/$DS-$tag/k$K
  write_meta "$out" cagra "$DS" "$K"
  "$ORIG_BIN/my_cagra" search "$INDEX_DIR/$DS/cagra/orig-d$deg/index" "$DATA_DIR/$DS/query.fvecs" "$IT" \
    "$out/it$IT.topk" "$sw" "$itopk" "$team" > "$out/it$IT.log" 2>&1
  python3 "$ORIG_TOOLS/cal_recall.py" --gt "$DATA_DIR/$DS/gt.ibin" --ret "$out/it$IT.topk" --k "$K" \
    --skip-head 3 --out "$out/it$IT.recall"
}

for v in $SEARCH_WIDTHS; do search "search_width-$v" "$BASE_DEGREE" "$v" "$BASE_ITOPK" "$BASE_TEAM_SIZE"; done
for v in $ITOPKS; do search "internal_topk-$v" "$BASE_DEGREE" "$BASE_SEARCH_WIDTH" "$v" "$BASE_TEAM_SIZE"; done
for v in $TEAM_SIZES; do search "team_size-$v" "$BASE_DEGREE" "$BASE_SEARCH_WIDTH" "$BASE_ITOPK" "$v"; done
for v in $DEGREES; do search "degree-$v" "$v" "$BASE_SEARCH_WIDTH" "$BASE_ITOPK" "$BASE_TEAM_SIZE"; done

echo "CAGRA parameter runs in $ORIG_OUT/cagra/$DS-*"
