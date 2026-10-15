#!/bin/bash
# Fetch every third-party source and build the systems used by the experiments. Each build writes
# its log to $W/build/logs/<name>.log; a failed build is reported at the end and does not stop the
# others. SONG, GANNS and BANG are compiled per parameter set; the lists below cover the defaults,
# and the graph-building scripts call song.sh, ganns.sh and bang.sh for other settings. Of the
# original-manuscript systems, the GGNN libraries with the sharded driver and the DiskANN tools are
# built here; the SONG and GANNS programs depend on the dataset and are built by song_orig.sh and
# experiments/original/build_{song,ganns}.sh when an experiment needs them.
#
# Usage: bash third_party/build/all.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"

GANNS_DIMS=${GANNS_DIMS:-"96 128 200 256 784 960"}   # DEEP, SIFT1M, GloVe200, NYTimes, MNIST8M, GIST
BANG_RS=${BANG_RS:-"32 64 96 128"}                    # graph degrees
SONG_CONFIGS=${SONG_CONFIGS:-"15:5:150:128"}          # SD:SHIFT:BUDGET:DIM

LOG=$W/build/logs
mkdir -p "$LOG"
FAILED=""
run() {
  local tag=$1; shift
  echo "$(date '+%F %T') $tag"
  if ! bash "$@" > "$LOG/$tag.log" 2>&1; then
    echo "  failed, see $LOG/$tag.log"
    FAILED="$FAILED $tag"
  fi
}

bash "$HERE/../fetch.sh"
run diskann "$HERE/diskann.sh"
run ggnn "$HERE/ggnn.sh"
for d in $GANNS_DIMS; do run "ganns_dim$d" "$HERE/ganns.sh" "$d"; done
for c in $SONG_CONFIGS; do
  IFS=: read -r sd sh budget dim <<< "$c"
  run "song_d${sd}_s${sh}_b${budget}_dim${dim}" "$HERE/song.sh" "$sd" "$sh" "$budget" "$dim"
done
for r in $BANG_RS; do run "bang_r$r" "$HERE/bang.sh" "$r"; done
run pilotann "$HERE/pilotann.sh"
run faiss "$HERE/faiss.sh"
run cuvs "$HERE/cuvs.sh"
run ggnn_orig "$HERE/ggnn_orig.sh"
run diskann_orig "$HERE/diskann_orig.sh"

if [ -n "$FAILED" ]; then
  echo "failed:$FAILED"
  exit 1
fi
echo "all builds done"
