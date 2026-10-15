// Print GPU properties for the platform description: SM count, L2 size, shared memory and threads per SM,
// memory size, bus width and peak memory bandwidth.
// usage: nvcc -o device_props device_props.cu && ./device_props > device_props.tsv
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  int mc = 0, bw = 0;
  cudaDeviceGetAttribute(&mc, cudaDevAttrMemoryClockRate, 0);
  cudaDeviceGetAttribute(&bw, cudaDevAttrGlobalMemoryBusWidth, 0);
  printf("name\t%s\ncc\t%d.%d\nsm_count\t%d\nl2_MB\t%.1f\nsmem_per_sm_KB\t%zu\nregs_per_sm\t%d\nmax_threads_per_sm\t%d\n"
         "mem_GB\t%.1f\nbus_width_bit\t%d\nmem_clock_MHz\t%d\npeak_bw_GBps\t%.0f\n",
         p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize / 1048576.0, p.sharedMemPerMultiprocessor / 1024,
         p.regsPerMultiprocessor, p.maxThreadsPerMultiProcessor, p.totalGlobalMem / 1e9, bw, mc / 1000,
         2.0 * mc * 1e3 * (bw / 8) / 1e9);
}
