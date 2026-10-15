#!/bin/bash
# Figure 16, step 1: build one NN-descent graph of SIFT1M and optimize it twice with CAGRA,
# with and without reverse edges (cagra_build_csr --ab). Both graphs therefore start from the same kNN graph.
# Needs cagra_build_csr linked against the libcuvs with the CUVS_CAGRA_NO_REVERSE patch.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

DS=${DS:-sift1M}
GD=${GD:-32}        # graph degree after optimize
IGD=${IGD:-64}      # degree of the NN-descent graph
NITER=${NITER:-20}  # NN-descent iterations
GRAPHS=${GRAPHS:-$INDEX_DIR/$DS/cagra/reverse_ab}
OUT=${OUT:-$OUT_DIR/reverse_edges}

mkdir -p "$GRAPHS" "$OUT"
"$BIN_DIR/cagra_build_csr" --ab "$DATA_DIR/$DS/base.fbin" "$GRAPHS" "$GD" "$IGD" "$NITER" 2>&1 | tee "$OUT/build.log"
cp "$GRAPHS/compare.txt" "$OUT/compare.txt"

# An unpatched libcuvs ignores CUVS_CAGRA_NO_REVERSE and returns the same graph twice
changed=$(sed -n 's/^edges_changed_set=\([0-9]*\).*/\1/p' "$OUT/compare.txt")
if [[ "${changed:-0}" == 0 ]]; then
  echo "the two graphs are identical: is libcuvs built with the CUVS_CAGRA_NO_REVERSE patch?" >&2
  exit 1
fi
echo "graphs in $GRAPHS/{with_reverse,no_reverse}, comparison in $OUT/compare.txt"
