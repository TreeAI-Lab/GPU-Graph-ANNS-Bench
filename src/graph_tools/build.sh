#!/bin/bash
# Builds sample_top32, edge_length_stats and cagra_build_csr into $BIN_DIR (see Makefile).
# Usage: bash build.sh [sample_top32|edge_length_stats|cagra_build_csr ...]
# Needs the cuVS build of third_party/build/cuvs.sh (with the CUVS_CAGRA_NO_REVERSE patch) and $ENV_CUVS.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

HERE=$(cd "$(dirname "$0")" && pwd)
JOBS=${JOBS:-3}

make -C "$HERE" -j "$JOBS" "$@"
ls -l "$BIN_DIR"/sample_top32 "$BIN_DIR"/edge_length_stats "$BIN_DIR"/cagra_build_csr 2>/dev/null || true
