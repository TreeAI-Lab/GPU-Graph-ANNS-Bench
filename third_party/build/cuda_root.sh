#!/bin/bash
# Print a CUDA toolkit root with the usual layout for a conda environment.
# Conda keeps the CUDA headers under targets/x86_64-linux/include, while CMake's FindCUDA module and
# torch.utils.cpp_extension look for include/ and lib64/ directly under the toolkit root. This makes a
# directory of symbolic links with that layout under $W/build/cuda-root/<environment>.
# Usage: CUDA_ROOT=$(bash third_party/build/cuda_root.sh <environment prefix>)
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

E=${1:?usage: cuda_root.sh <environment prefix>}
R=$W/build/cuda-root/$(basename "$E")
mkdir -p "$R"
ln -sfn "$E/bin" "$R/bin"
ln -sfn "$E/targets/x86_64-linux/include" "$R/include"
ln -sfn "$E/lib" "$R/lib64"
ln -sfn "$E/targets" "$R/targets"
if [ -d "$E/nvvm" ]; then ln -sfn "$E/nvvm" "$R/nvvm"; fi
echo "$R"
