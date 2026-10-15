#!/bin/bash
# Build the GANNS programs of the original runs (ganns-orig with ganns_orig.patch) for one dimension and
# distance, following the GANNS README: generate_template.sh, then generate_build_instances.sh and
# generate_query_instances.sh <dim> <metric>.
#
# The per-stage clocks, cal_count, path_length, the graph analysis (average degree, connected
# components) and the 64-bit offsets needed for MNIST8M come from ganns_orig.patch; generate_template.sh
# expands the macro files of the patched sources into template/, from which both programs are built.
#
# Usage:  bash build_ganns.sh <dim> <l2|cos>
# Output: $BIN_DIR/original/ganns/build_<dim>_<metric>, query_<dim>_<metric>
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

DIM=${1:?usage: build_ganns.sh <dim> <l2|cos>}
METRIC=${2:?usage: build_ganns.sh <dim> <l2|cos>}
SRC=${GANNS_SRC:-$W/src/ganns-orig}

export PATH=$ENV_BUILD/bin:$PATH
WORK=$W/build/original/ganns/${DIM}_$METRIC
rm -rf "$WORK" && mkdir -p "$WORK"
cp -r "$SRC/." "$WORK/" && rm -rf "$WORK/.git"
cd "$WORK"
grep -q "SearchDevice_B" kernel_search_nsw.h || { echo "$SRC lacks ganns_orig.patch" >&2; exit 1; }
bash generate_template.sh > generate_template.log 2>&1
bash generate_build_instances.sh "$DIM" "$METRIC" > build_instance.log 2>&1
bash generate_query_instances.sh "$DIM" "$METRIC" > query_instance.log 2>&1

mkdir -p "$ORIG_BIN/ganns"
for prog in build query; do
  exe=${prog}_${DIM}_$METRIC
  [ -x "$exe" ] || { echo "$exe was not built, see $WORK/${prog}_instance.log" >&2; exit 1; }
  cp "$exe" "$ORIG_BIN/ganns/$exe"
done
ls -l "$ORIG_BIN/ganns/build_${DIM}_$METRIC" "$ORIG_BIN/ganns/query_${DIM}_$METRIC"
