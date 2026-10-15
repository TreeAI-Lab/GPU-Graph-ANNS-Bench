#!/bin/bash
# Build the baseline programs into $BIN_DIR.
# Usage: bash src/baselines/build.sh [faiss|cuvs|all]
#   faiss: needs Faiss built by third_party/build/faiss.sh and the environment $ENV_FAISS
#   cuvs:  needs cuVS built by third_party/build/cuvs.sh and the environment $ENV_CUVS
# Make variables can be overridden on the command line after the target, e.g.
#   bash src/baselines/build.sh cuvs CUVS_DEPS=/path/to/_deps
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

HERE=$(cd "$(dirname "$0")" && pwd)
TARGET=${1:-all}
shift || true
JOBS=${JOBS:-4}

make -C "$HERE" -j"$JOBS" "$TARGET" \
  W="$W" BIN_DIR="$BIN_DIR" ENV_FAISS="$ENV_FAISS" ENV_CUVS="$ENV_CUVS" "$@"
