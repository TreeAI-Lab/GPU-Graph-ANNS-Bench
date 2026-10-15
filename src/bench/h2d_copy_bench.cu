// Host-to-device copy time of one file (base vectors or a graph index).
// The file is read into a pageable malloc buffer (not cudaMallocHost), its pages are touched,
// and then `warmup` untimed and `reps` timed cudaMemcpy calls copy it to the GPU (CUDA events).
//
// Usage: h2d_copy_bench <dataset> <algo> <config> <file> [warmup=2] [reps=5]
// Output: one tab-separated line on stdout:
//   dataset algo config file bytes median_ms min_ms max_ms mean_ms median_GBps trials_ms
// (trials_ms: the timed copies, comma-separated). Progress goes to stderr.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <sys/stat.h>

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__,        \
              __LINE__, cudaGetErrorString(_err));                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int bench_file(const char *dataset, const char *algo, const char *config,
                      const char *path, int warmup, int reps) {
  struct stat st;
  if (stat(path, &st) != 0) {
    fprintf(stderr, "stat failed: %s\n", path);
    return 1;
  }
  const size_t nbytes = static_cast<size_t>(st.st_size);
  if (nbytes == 0) {
    fprintf(stderr, "empty file: %s\n", path);
    return 1;
  }

  fprintf(stderr, "read %s (%.2f MiB) into pageable malloc...\n", path,
          nbytes / (1024.0 * 1024.0));
  FILE *fp = fopen(path, "rb");
  if (!fp) {
    fprintf(stderr, "fopen failed: %s\n", path);
    return 1;
  }
  void *host = malloc(nbytes);
  if (!host) {
    fprintf(stderr, "malloc failed for %zu bytes\n", nbytes);
    fclose(fp);
    return 1;
  }
  size_t got = 0;
  while (got < nbytes) {
    size_t chunk = nbytes - got;
    if (chunk > (64u << 20)) chunk = (64u << 20);
    size_t r = fread(static_cast<char *>(host) + got, 1, chunk, fp);
    if (r == 0) {
      fprintf(stderr, "short read %zu / %zu\n", got, nbytes);
      free(host);
      fclose(fp);
      return 1;
    }
    got += r;
  }
  fclose(fp);

  // Fault pages in so the timed cudaMemcpy is the pageable copy itself.
  volatile unsigned char sink = 0;
  auto *bytes = static_cast<unsigned char *>(host);
  for (size_t i = 0; i < nbytes; i += 4096) sink = static_cast<unsigned char>(sink + bytes[i]);
  if (nbytes) sink = static_cast<unsigned char>(sink + bytes[nbytes - 1]);

  void *device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, nbytes));

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(cudaMemcpy(device, host, nbytes, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> ms(reps);
  for (int i = 0; i < reps; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(device, host, nbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms[i], start, stop));
    fprintf(stderr, "  trial %d: %.3f ms\n", i, ms[i]);
  }

  std::vector<float> sorted = ms;
  std::sort(sorted.begin(), sorted.end());
  float median = sorted[reps / 2];
  float min_ms = sorted.front();
  float max_ms = sorted.back();
  double sum = 0;
  for (float v : ms) sum += v;
  float mean = static_cast<float>(sum / reps);
  double gbps = (nbytes / 1e9) / (median / 1e3);

  // dataset algo config file bytes median_ms min_ms max_ms mean_ms median_GBps trials
  printf("%s\t%s\t%s\t%s\t%zu\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t", dataset, algo,
         config, path, nbytes, median, min_ms, max_ms, mean, gbps);
  for (int i = 0; i < reps; ++i) {
    if (i) printf(",");
    printf("%.3f", ms[i]);
  }
  printf("\n");
  fflush(stdout);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(device);
  free(host);
  (void)sink;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr,
            "usage: %s <dataset> <algo> <config> <file> [warmup] [reps]\n",
            argv[0]);
    return 2;
  }
  int warmup = argc > 5 ? std::atoi(argv[5]) : 2;
  int reps = argc > 6 ? std::atoi(argv[6]) : 5;
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  fprintf(stderr, "GPU %s\n", prop.name);
  return bench_file(argv[1], argv[2], argv[3], argv[4], warmup, reps);
}
