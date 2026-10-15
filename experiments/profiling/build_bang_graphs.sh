#!/bin/bash
# BANG graphs of Table 9 and the BANG programs for them.
#   1. DiskANN build_disk_index (Vamana graph with PQ), R = the matched maximum degree of the paper:
#      DEEP1M 32, GIST 128; L = 100; -B 4 -M 16 (required by build_disk_index); QD not given, so DiskANN derives
#      the PQ code size from B (96 bytes per vector for DEEP1M, 512 for GIST).
#   2. disk_index_to_bang.py: <ds>_disk.index -> <ds>_disk.bin and <ds>_disk_metadata.bin (GIST nodes span two
#      sectors, which BANG's own bang_preprocess.py cannot read).
#   3. split_pq_pivots.py: BANG reads the PQ pivots in the older three-file layout; the three files go to
#      bang_format/, with links to the other three files.
#   4. third_party/build/bang.sh <R>: BANG compiled with MAX_R = R.
# usage: bash build_bang_graphs.sh [deep1M] [gist]     (no argument: both)
# output: $INDEX_DIR/<ds>/bang/r<R>_l100_b4/ (DiskANN files, build.log, convert.log) and its bang_format/<ds>_*,
#   the prefix that run_point.sh passes to BANG; $W/build/bang/r<R>/bang_search
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
BDI=$W/build/diskann/apps/build_disk_index   # third_party/build/diskann.sh
TOOLS=$BENCH/src/graph_tools
PY=${PY:-$ENV_BUILD/bin/python}
THREADS=${THREADS:-36}
L=100; B=4; M=16
export LD_LIBRARY_PATH=$ENV_BUILD/lib:${LD_LIBRARY_PATH:-}
DATASETS=("$@")
[[ ${#DATASETS[@]} -gt 0 ]] || DATASETS=(deep1M gist)
for ds in "${DATASETS[@]}"; do
  case "$ds" in
    deep1M) DIM=96; R=32 ;;
    gist) DIM=960; R=128 ;;
    *) echo "unknown dataset $ds (deep1M gist)" >&2; exit 1 ;;
  esac
  O=$INDEX_DIR/$ds/bang/r${R}_l${L}_b${B}
  mkdir -p "$O/bang_format"
  echo "bang $ds R=$R build $(date '+%F %T')"
  "$ENV_BUILD/bin/time" -v "$BDI" --data_type float --dist_fn l2 --data_path "$DATA_DIR/$ds/base.fbin" \
    --index_path_prefix "$O/$ds" -R "$R" -L "$L" -B "$B" -M "$M" -T "$THREADS" > "$O/build.log" 2>&1
  "$PY" "$TOOLS/disk_index_to_bang.py" "$O/${ds}_disk.index" "$O/${ds}_disk.bin" "$DIM" 2 "$R" > "$O/convert.log" 2>&1
  "$PY" "$TOOLS/split_pq_pivots.py" "$O/${ds}_pq_pivots.bin" "$O/bang_format/$ds"
  for f in _pq_compressed.bin _disk.bin _disk_metadata.bin; do ln -sfn "../$ds$f" "$O/bang_format/$ds$f"; done
  bash "$BENCH/third_party/build/bang.sh" "$R"
  echo "bang $ds done $(date '+%F %T')"
done
