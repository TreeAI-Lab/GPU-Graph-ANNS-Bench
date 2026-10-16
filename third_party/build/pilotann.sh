#!/bin/bash
# Build PilotANN ($W/src/pilotann, patched) as an editable install into the pilot environment.
# setup.py imports torch, so pip runs without build isolation. The GPU architectures come from
# TORCH_CUDA_ARCH_LIST (setup.py passes no -gencode flags).
#
# PILOT_NO_KERNEL_STATS=1 builds the GPU traversal kernel without the search statistics of the patch (cycle
# timers, counters and the barriers they need), so that it is the upstream kernel; experiments/profiling uses this
# build for the hardware counters. The default build keeps them (Figures 7, 9 and 12).
#
# Usage:  [PILOT_NO_KERNEL_STATS=1] [TORCH_CUDA_ARCH_LIST=8.6] bash third_party/build/pilotann.sh
# Output: the pilot_ann package and its pilot_ann/extension module, importable from $ENV_PILOT
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.0;8.0"}
# nvcc adds NVCC_APPEND_FLAGS to every compilation (setup.py passes no extra nvcc flags)
if [ "${PILOT_NO_KERNEL_STATS:-0}" = 1 ]; then
  export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -DPILOT_NO_KERNEL_STATS"
fi
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
echo "kernel statistics: $([ "${PILOT_NO_KERNEL_STATS:-0}" = 1 ] && echo off || echo on)"
