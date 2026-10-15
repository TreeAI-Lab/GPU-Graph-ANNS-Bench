#!/bin/bash
# Build PilotANN ($W/src/pilotann, patched) as an editable install into the pilot environment.
# setup.py imports torch, so pip runs without build isolation. The GPU architectures come from
# TORCH_CUDA_ARCH_LIST (setup.py passes no -gencode flags).
#
# Usage:  bash third_party/build/pilotann.sh
# Output: the pilot_ann package and its pilot_ann/extension module, importable from $ENV_PILOT
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.0;8.0"}
E=$ENV_PILOT

# torch.utils.cpp_extension looks for $CUDA_HOME/include and lib64; cuda_root.sh gives the environment that layout
CUDA_HOME=$(bash "$(cd "$(dirname "$0")" && pwd)/cuda_root.sh" "$E")
export PATH=$E/bin:$PATH CUDA_HOME
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=$E/bin/x86_64-conda-linux-gnu-g++
cd "$W/src/pilotann"
# Remove objects and extension modules left by an earlier build with another Python or CUDA.
rm -rf build PilotANN.egg-info pilot_ann/extension.*.so
"$E/bin/python" -m pip install -q --no-build-isolation -e .
"$E/bin/python" -c "import pilot_ann, pilot_ann.extension as x; print('pilot_ann ok', x.__file__)"
