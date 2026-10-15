#!/bin/bash
# Create the conda environments under $W/envs. Usage: bash environment/create_envs.sh [--locked] [names...]
# --locked installs the exact package builds of the paper runs from environment/locks/.
# CONDA selects the tool: conda (default) or micromamba.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
CONDA=${CONDA:-conda}
LOCKED=0
if [ "${1:-}" = "--locked" ]; then LOCKED=1; shift; fi
[ $# -gt 0 ] || set -- build cagra pilot faiss-build cuvs-build
for name in "$@"; do
  prefix=$W/envs/$name
  if [ $LOCKED = 1 ]; then
    "$CONDA" create -y -p "$prefix" --file "$HERE/locks/$name.txt"
  elif [ "$(basename "$CONDA")" = micromamba ]; then
    "$CONDA" create -y -p "$prefix" -f "$HERE/$name.yml"
  else
    "$CONDA" env create -y -p "$prefix" -f "$HERE/$name.yml"
  fi
done
for name in "$@"; do
  if [ "$name" = pilot ]; then
    "$W/envs/pilot/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
  fi
done
