/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Build-only variant of the cuVS CAGRA example, used for graphs whose base set does not fit NN-descent in GPU
// memory (DEEP100M on an 80 GB GPU): CAGRA with the IVF-PQ graph builder, saved without the dataset.

#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <utility>
#include <sys/stat.h>    // fstat, stat, struct stat
#include <sys/mman.h>    // mmap, munmap, PROT_READ, MAP_SHARED, MAP_FAILED
#include <fcntl.h>       // open
#include <unistd.h>
#include <unordered_set>
#include <cuda_fp16.h>
#include <cassert>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>

#include <cuvs/neighbors/cagra.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"


using HostAccessor = raft::host_device_accessor<std::experimental::default_accessor<const float>, raft::memory_type::host>;
using HostMdspanT = raft::mdspan<const float, raft::matrix_extent<int64_t>, raft::row_major, HostAccessor>;

// Reads an fvecs file into a row-major float array.
std::vector<float> read_fvecs(const std::string& filename, int64_t& num_vectors, int64_t& dim) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  std::vector<float> data;
  int read_dim;
  
  // The first int is the dimension
  in.read(reinterpret_cast<char*>(&read_dim), sizeof(int));
  dim = read_dim;
  
  in.seekg(0, std::ios::beg);
  
  // Number of vectors from the file size
  in.seekg(0, std::ios::end);
  size_t filesize = in.tellg();
  in.seekg(0, std::ios::beg);
  
  if (filesize % (dim * sizeof(float) + sizeof(int)) != 0) {
    throw std::runtime_error("Invalid fvecs file format");
  }
  
  num_vectors = filesize / (dim * sizeof(float) + sizeof(int));
  data.resize(num_vectors * dim);
  
  std::vector<float> buffer(dim);
  for (int64_t i = 0; i < num_vectors; i++) {
    in.read(reinterpret_cast<char*>(&read_dim), sizeof(int));
    if (read_dim != dim) {
      throw std::runtime_error("Inconsistent vector dimensions in file");
    }
    
    in.read(reinterpret_cast<char*>(buffer.data()), dim * sizeof(float));
    std::copy(buffer.begin(), buffer.end(), data.begin() + i * dim);
  }
  
  return data;
}

struct MappedFbin {
  int fd = -1;
  void* map = MAP_FAILED;
  size_t bytes = 0;
  const float* data = nullptr;
  int64_t n = 0;
  int64_t dim = 0;

  MappedFbin() = default;
  MappedFbin(const MappedFbin&) = delete;
  MappedFbin& operator=(const MappedFbin&) = delete;
  MappedFbin(MappedFbin&& o) noexcept { *this = std::move(o); }
  MappedFbin& operator=(MappedFbin&& o) noexcept {
    if (this == &o) return *this;
    if (map != MAP_FAILED) munmap(map, bytes);
    if (fd >= 0) close(fd);
    fd = o.fd;
    map = o.map;
    bytes = o.bytes;
    data = o.data;
    n = o.n;
    dim = o.dim;
    o.fd = -1;
    o.map = MAP_FAILED;
    o.bytes = 0;
    o.data = nullptr;
    o.n = 0;
    o.dim = 0;
    return *this;
  }

  ~MappedFbin() {
    if (map != MAP_FAILED) munmap(map, bytes);
    if (fd >= 0) close(fd);
  }
};

MappedFbin map_fbin(const std::string& filename) {
  MappedFbin m;
  m.fd = open(filename.c_str(), O_RDONLY);
  if (m.fd < 0) throw std::runtime_error("Failed to open file: " + filename);
  struct stat st;
  if (fstat(m.fd, &st) != 0) throw std::runtime_error("fstat failed: " + filename);
  m.bytes = static_cast<size_t>(st.st_size);
  m.map = mmap(nullptr, m.bytes, PROT_READ, MAP_PRIVATE, m.fd, 0);
  if (m.map == MAP_FAILED) throw std::runtime_error("mmap failed: " + filename);
  madvise(m.map, m.bytes, MADV_SEQUENTIAL);
  if (m.bytes < 8) throw std::runtime_error("fbin too small: " + filename);
  int32_t header[2];
  std::memcpy(header, m.map, sizeof(header));
  m.n = header[0];
  m.dim = header[1];
  const size_t need = 8ull + static_cast<size_t>(m.n) * static_cast<size_t>(m.dim) * sizeof(float);
  if (m.n <= 0 || m.dim <= 0 || need != m.bytes) {
    throw std::runtime_error("fbin header does not match file size: " + filename);
  }
  m.data = reinterpret_cast<const float*>(static_cast<char*>(m.map) + 8);
  return m;
}

struct MemPoll {
  std::atomic<bool> stop{false};
  std::mutex mu;
  std::ofstream out;
  std::thread th;
  std::chrono::steady_clock::time_point t0;
  double peak_gpu_mib = 0;
  double peak_rss_mib = 0;
  double peak_hwm_mib = 0;

  explicit MemPoll(const std::string& path)
    : out(path), t0(std::chrono::steady_clock::now()) {
    out << "t_s\tstage\tgpu_used_mib\tgpu_free_mib\trss_mib\thwm_mib\n";
    out.flush();
    th = std::thread([this] { loop(); });
  }

  ~MemPoll() { finish(); }

  void finish() {
    bool expected = false;
    if (stop.compare_exchange_strong(expected, true)) {
      if (th.joinable()) th.join();
    }
  }

  static void read_host(double& rss_mib, double& hwm_mib) {
    rss_mib = 0;
    hwm_mib = 0;
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
      if (line.rfind("VmRSS:", 0) == 0) rss_mib = std::atof(line.c_str() + 6) / 1024.0;
      if (line.rfind("VmHWM:", 0) == 0) hwm_mib = std::atof(line.c_str() + 6) / 1024.0;
    }
  }

  static void read_gpu(double& used_mib, double& free_mib) {
    used_mib = -1;
    free_mib = -1;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
      used_mib = (total_b - free_b) / 1048576.0;
      free_mib = free_b / 1048576.0;
      return;
    }
    FILE* p = popen(
      "nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits", "r");
    if (!p) return;
    int used = 0, free = 0;
    if (fscanf(p, "%d, %d", &used, &free) == 2) {
      used_mib = used;
      free_mib = free;
    }
    pclose(p);
  }

  void sample(const char* stage) {
    double used = 0, free = 0, rss = 0, hwm = 0;
    read_gpu(used, free);
    read_host(rss, hwm);
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lock(mu);
    if (used > peak_gpu_mib) peak_gpu_mib = used;
    if (rss > peak_rss_mib) peak_rss_mib = rss;
    if (hwm > peak_hwm_mib) peak_hwm_mib = hwm;
    out << t << "\t" << stage << "\t" << used << "\t" << free << "\t" << rss << "\t" << hwm << "\n";
    out.flush();
    std::cout << "[mem] t=" << t << "s stage=" << stage
              << " gpu_used=" << used << " MiB rss=" << rss << " MiB hwm=" << hwm << " MiB"
              << std::endl;
  }

  void loop() {
    while (!stop.load()) {
      sample("poll");
      for (int i = 0; i < 20 && !stop.load(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }
};

// Builds a graph-only CAGRA index with the IVF-PQ graph builder and writes it to index_file.
void cagra_build_and_save(raft::device_resources const& dev_resources,
                         HostMdspanT dataset,
                         const std::string& index_file,
                         int degree,
                         MemPoll& mem)
{
  using namespace cuvs::neighbors;

  cagra::index_params index_params;

  std::cout << "Building CAGRA index with IVF-PQ (NN-descent does not fit on this GPU)" << std::endl;
  index_params.graph_degree = degree;
  index_params.intermediate_graph_degree = degree * 2;
  index_params.metric = cuvs::distance::DistanceType::L2Expanded;
  // Graph only. Attaching the 100M dataset would need another ~38 GiB on the device
  // after the IVF-PQ working set is freed, and the other deep graphs are stored graph-only.
  index_params.attach_dataset_on_build = false;

  auto pq_params = cagra::graph_build_params::ivf_pq_params(
    dataset.extents(), cuvs::distance::DistanceType::L2Expanded);
  pq_params.build_params.conservative_memory_allocation = true;
  index_params.graph_build_params = pq_params;

  std::cout << "IVF-PQ n_lists=" << pq_params.build_params.n_lists
            << " pq_dim=" << pq_params.build_params.pq_dim
            << " pq_bits=" << pq_params.build_params.pq_bits
            << " kmeans_n_iters=" << pq_params.build_params.kmeans_n_iters
            << " kmeans_trainset_fraction=" << pq_params.build_params.kmeans_trainset_fraction
            << " n_probes=" << pq_params.search_params.n_probes
            << " refinement_rate=" << pq_params.refinement_rate
            << " graph_degree=" << index_params.graph_degree
            << " intermediate_graph_degree=" << index_params.intermediate_graph_degree
            << std::endl;

  mem.sample("before_build");
  auto start = std::chrono::high_resolution_clock::now();
  auto index = cagra::build(dev_resources, index_params, dataset);
  raft::resource::sync_stream(dev_resources);
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  mem.sample("after_build");

  std::cout << "Built index type: " << typeid(index).name() << std::endl;
  std::cout << "CAGRA index built in " << duration << " ms" << std::endl;
  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  std::cout << "Saving index to: " << index_file << std::endl;
  auto save0 = std::chrono::high_resolution_clock::now();
  cagra::serialize(dev_resources, index_file, index);
  raft::resource::sync_stream(dev_resources);
  auto save1 = std::chrono::high_resolution_clock::now();
  mem.sample("after_serialize");
  auto save_ms = std::chrono::duration_cast<std::chrono::milliseconds>(save1 - save0).count();
  std::cout << "Index saved successfully in " << save_ms << " ms" << std::endl;
  std::cout << "BUILD_MS " << duration << std::endl;
  std::cout << "SERIALIZE_MS " << save_ms << std::endl;
  std::cout << "PEAK_GPU_MIB " << mem.peak_gpu_mib << std::endl;
  std::cout << "PEAK_RSS_MIB " << mem.peak_rss_mib << std::endl;
  std::cout << "PEAK_HWM_MIB " << mem.peak_hwm_mib << std::endl;
}

int main(int argc, char** argv)
{
  if (argc < 5 || std::string(argv[1]) != "build") {
    std::cerr << "Usage: " << argv[0]
              << " build <base_file.fbin|fvecs> <index_file> <degree> [mem_tsv]" << std::endl;
    return 1;
  }

  raft::device_resources dev_resources;

  // Pool memory resource with a 3 GiB initial size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 3*1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  std::string base_file = argv[2];
  std::string index_file = argv[3];
  int degree = std::stoi(argv[4]);
  std::string mem_tsv = argc > 5 ? argv[5] : (index_file + ".mem.tsv");
  MemPoll mem(mem_tsv);
  mem.sample("start");

  const bool is_fbin = base_file.size() >= 5 &&
                       base_file.compare(base_file.size() - 5, 5, ".fbin") == 0;
  try {
    if (is_fbin) {
      std::cout << "Mapping fbin: " << base_file << std::endl;
      MappedFbin mapped = map_fbin(base_file);
      std::cout << "Base dataset: " << mapped.n << " vectors of dimension " << mapped.dim
                << " (" << (mapped.bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB file)"
                << std::endl;
      mem.sample("after_mmap");
      HostMdspanT host_md(mapped.data, mapped.n, mapped.dim);
      cagra_build_and_save(dev_resources, host_md, index_file, degree, mem);
    } else {
      std::cout << "Reading base vectors from: " << base_file << std::endl;
      int64_t n_samples, n_dim;
      std::vector<float> h_base = read_fvecs(base_file, n_samples, n_dim);
      std::cout << "Base dataset: " << n_samples << " vectors of dimension " << n_dim << std::endl;
      mem.sample("after_read");
      HostMdspanT host_md(h_base.data(), n_samples, n_dim);
      cagra_build_and_save(dev_resources, host_md, index_file, degree, mem);
    }
  } catch (const std::exception& e) {
    std::cerr << "Build failed: " << e.what() << std::endl;
    mem.sample("failed");
    return 1;
  }
  mem.sample("done");
  mem.finish();

  std::cout << "CAGRA processing completed." << std::endl;
  return 0;
}
