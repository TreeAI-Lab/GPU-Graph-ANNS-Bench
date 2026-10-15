#!/bin/bash
# Download the raw files of the evaluated datasets into $DATA_DIR/raw.
# Usage: bash datasets/download.sh sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M
# Deep1B prefixes are fetched by HTTP range requests, so only the needed part of the 1B file is downloaded.
# Set DEEP1B_URL or HF_ENDPOINT to use a mirror.
set -euo pipefail

DATA_DIR=${DATA_DIR:-data}
ANN=${ANN_BENCHMARKS_URL:-http://ann-benchmarks.com}
DEEP1B=${DEEP1B_URL:-https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP}
HF=${HF_ENDPOINT:-https://huggingface.co}
RAW=$DATA_DIR/raw
mkdir -p "$RAW"

get() {  # get <file> <url>: resumable download
  echo "downloading $1"
  curl -fL --retry 5 -C - -o "$RAW/$1" "$2"
}

deep1b_prefix() {  # deep1b_prefix <n vectors> <file>: first n vectors of base.1B.fbin, header set to (n, 96)
  local n=$1 out=$RAW/$2 last=$((8 + $1 * 96 * 4 - 1))
  local have=0
  [ -f "$out" ] && have=$(wc -c < "$out")
  if [ "$have" -le "$last" ]; then
    echo "downloading the first $n vectors of Deep1B"
    curl -fL --retry 5 -r "$have-$last" "$DEEP1B/base.1B.fbin" >> "$out"
  fi
  python3 - "$out" "$n" <<'EOF'
import struct, sys
with open(sys.argv[1], 'r+b') as f:
    f.write(struct.pack('<ii', int(sys.argv[2]), 96))
EOF
  [ -f "$RAW/deep1b_query.public.10K.fbin" ] || get deep1b_query.public.10K.fbin "$DEEP1B/query.public.10K.fbin"
}

for name in "$@"; do
  case $name in
    sift1M) get sift-128-euclidean.hdf5 "$ANN/sift-128-euclidean.hdf5" ;;
    gist) get gist-960-euclidean.hdf5 "$ANN/gist-960-euclidean.hdf5" ;;
    glove200) get glove-200-angular.hdf5 "$ANN/glove-200-angular.hdf5" ;;
    nytimes|nytimes-full) get nytimes-256-angular.hdf5 "$ANN/nytimes-256-angular.hdf5" ;;
    deep1M) deep1b_prefix 1000000 deep1b_first1M.fbin ;;
    deep10M) deep1b_prefix 10000000 deep1b_first10M.fbin ;;
    deep100M) deep1b_prefix 100000000 deep1b_first100M.fbin ;;
    deep1M-annb|deep10M-annb) get deep-image-96-angular.hdf5 "$ANN/deep-image-96-angular.hdf5" ;;
    mnist8M) get MNIST8M_data.h5 "$HF/datasets/MarveenLee/MNIST8M/resolve/main/MNIST8M_data.h5" ;;
    *) echo "unknown dataset $name" >&2; exit 1 ;;
  esac
done
