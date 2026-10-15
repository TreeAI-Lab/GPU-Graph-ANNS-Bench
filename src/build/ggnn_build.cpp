// GGNN build driver: reads fvecs, builds the graph and stores it as <out_dir>/part_0.ggnn, using only the public
// GGNN interface. GGNN_BUILD_SECONDS is the time of ggnn.build(), which includes uploading the base vectors to
// the GPU and copying the graph back.
// Usage: ggnn_build <base.fvecs> <out_dir> <k_build> <tau_build> <refinement_iterations> <gpu_id>
#include <ggnn/base/ggnn.cuh>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <vector>

int main(int argc, char** argv)
{
  if (argc != 7) { std::fprintf(stderr, "usage: %s base.fvecs out_dir k_build tau_build refine gpu\n", argv[0]); return 1; }
  using GGNN = ggnn::GGNN<int32_t, float>;
  const std::filesystem::path out_dir{argv[2]};
  std::filesystem::create_directories(out_dir);
  GGNN ggnn{};
  ggnn.setWorkingDirectory(out_dir);
  ggnn.setGPUs(std::vector<int>{std::atoi(argv[6])});
  ggnn.setBase(ggnn::GenericDataset::load(argv[1]));
  const auto t0 = std::chrono::steady_clock::now();
  ggnn.build(static_cast<uint32_t>(std::atoi(argv[3])), static_cast<float>(std::atof(argv[4])),
             static_cast<uint32_t>(std::atoi(argv[5])));
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ggnn.store();
  std::printf("GGNN_BUILD_SECONDS %.3f\n", secs);
  return 0;
}
