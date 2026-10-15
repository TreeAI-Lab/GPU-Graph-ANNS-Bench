// Minimal kernel for run_smoke.sh: reads device memory at random indices, like the memory accesses of graph search.
// It only checks that Nsight Compute can read the GPU performance counters.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void gather(const float *x, const int *idx, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = x[idx[i]] * 2.0f;
}

int main() {
  const int n = 1 << 24;
  float *x, *out;
  int *idx;
  cudaMalloc(&x, n * sizeof(float));
  cudaMalloc(&out, n * sizeof(float));
  cudaMallocManaged(&idx, n * sizeof(int));
  unsigned s = 12345;
  for (int i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    idx[i] = s % n;
  }
  cudaMemset(x, 0, n * sizeof(float));
  // Two launches; the profiler skips the first, as for the search kernels
  for (int r = 0; r < 2; ++r) gather<<<(n + 255) / 256, 256>>>(x, idx, out, n);
  cudaError_t e = cudaDeviceSynchronize();
  printf("gather done: %s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
