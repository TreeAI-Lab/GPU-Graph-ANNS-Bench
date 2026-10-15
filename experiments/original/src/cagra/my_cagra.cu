// CAGRA build and search driver of the original manuscript (cuVS). Built by ../../build_cagra.sh.
//
//   my_cagra build  <base.fvecs> <index_file> <degree>
//   my_cagra search <index_file> <query.fvecs> [iterations=10] [output_file] <search_width> <internal_topk> <team_size>
//
// build: graph degree <degree>, intermediate degree 2 * <degree>, L2 distance; prints the build time.
// search: single-CTA search of all queries in one call, k = 10; writes the neighbor ids of every query
// to <output_file> after three header lines starting with '#'.

#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include <cuda_fp16.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuvs/neighbors/cagra.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

using HostAccessor = raft::host_device_accessor<std::experimental::default_accessor<const float>,
                                                raft::memory_type::host>;
using HostMdspanT = raft::mdspan<const float, raft::matrix_extent<int64_t>, raft::row_major, HostAccessor>;

std::vector<float> read_fvecs(const std::string& filename, int64_t& num_vectors, int64_t& dim)
{
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) { throw std::runtime_error("Failed to open file: " + filename); }

  std::vector<float> data;
  int read_dim;

  // the dimension of the first vector fixes the record size
  in.read(reinterpret_cast<char*>(&read_dim), sizeof(int));
  dim = read_dim;

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
    if (read_dim != dim) { throw std::runtime_error("Inconsistent vector dimensions in file"); }

    in.read(reinterpret_cast<char*>(buffer.data()), dim * sizeof(float));
    std::copy(buffer.begin(), buffer.end(), data.begin() + i * dim);
  }

  return data;
}

void cagra_build_and_save(raft::device_resources const& dev_resources,
                          HostMdspanT dataset,
                          const std::string& index_file,
                          int degree)
{
  using namespace cuvs::neighbors;

  cagra::index_params index_params;

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  index_params.graph_degree              = degree;
  index_params.intermediate_graph_degree = degree * 2;
  index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;

  auto start = std::chrono::high_resolution_clock::now();
  auto index = cagra::build(dev_resources, index_params, dataset);
  std::cout << "Built index type: " << typeid(index).name() << std::endl;

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "CAGRA index built in " << duration << " ms" << std::endl;
  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  std::cout << "Saving index to: " << index_file << std::endl;
  cagra::serialize(dev_resources, index_file, index);
  std::cout << "Index saved successfully" << std::endl;
}

void cagra_load_and_search(raft::device_resources const& dev_resources,
                           raft::device_matrix_view<const float, int64_t> queries,
                           const std::string& index_file,
                           int iterations                 = 10,
                           const std::string& output_file = "",
                           int search_width               = 100,
                           int internal_topk              = 32,
                           int team_size                  = 32)
{
  using namespace cuvs::neighbors;

  int64_t topk      = 10;
  int64_t n_queries = queries.extent(0);

  auto neighbors = raft::make_device_matrix<uint32_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  std::cout << "Loading index from: " << index_file << std::endl;
  auto start_load = std::chrono::high_resolution_clock::now();

  cagra::index<float, unsigned int> index(dev_resources);
  cagra::deserialize(dev_resources, index_file, &index);
  auto dataset_rows = index.dataset().extent(0);
  auto dataset_dims = index.dataset().extent(1);
  printf("n_dataset: %ld, dim: %ld\n", dataset_rows, dataset_dims);
  auto end_load  = std::chrono::high_resolution_clock::now();
  auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count();

  std::cout << "deserialize time " << load_time << " ms" << std::endl;
  std::cout << "CAGRA index has " << index.size() << " vectors" << std::endl;
  std::cout << "CAGRA graph has degree " << index.graph_degree() << ", graph size ["
            << index.graph().extent(0) << ", " << index.graph().extent(1) << "]" << std::endl;

  cagra::search_params search_params;
  search_params.itopk_size     = internal_topk;
  search_params.search_width   = search_width;
  search_params.team_size      = team_size;
  search_params.max_iterations = iterations;
  search_params.algo           = cagra::search_algo::SINGLE_CTA;
  search_params.persistent     = false;
  std::cout << "Using search iterations: " << iterations << std::endl;

  cagra::search(dev_resources, search_params, index, queries, neighbors.view(), distances.view());

  raft::resource::sync_stream(dev_resources);

  if (!output_file.empty()) {
    std::cout << "Writing search results to file: " << output_file << std::endl;

    std::vector<uint32_t> h_neighbors(n_queries * topk);
    std::vector<float> h_distances(n_queries * topk);

    raft::update_host(h_neighbors.data(), neighbors.data_handle(), n_queries * topk,
                      raft::resource::get_cuda_stream(dev_resources));
    raft::update_host(h_distances.data(), distances.data_handle(), n_queries * topk,
                      raft::resource::get_cuda_stream(dev_resources));
    raft::resource::sync_stream(dev_resources);

    std::ofstream outfile(output_file);
    if (!outfile) {
      std::cerr << "Error: Could not open output file: " << output_file << std::endl;
    } else {
      outfile << "# CAGRA search results" << std::endl;
      outfile << "# Queries: " << n_queries << ", K: " << topk << ", Iterations: " << iterations << std::endl;
      outfile << "# Format: neighbor_1 neighbor_2 ... neighbor_k" << std::endl;

      for (int64_t i = 0; i < n_queries; i++) {
        for (int64_t j = 0; j < topk; j++) {
          outfile << h_neighbors[i * topk + j];
          if (j < topk - 1) outfile << " ";
        }
        outfile << std::endl;
      }

      std::cout << "Results successfully written to file" << std::endl;
    }
  }
}

int main(int argc, char** argv)
{
  if (argc < 3) {
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  Build index: " << argv[0] << " build <base_file.fvecs> <index_file> <degree>" << std::endl;
    std::cerr << "  Search index: " << argv[0]
              << " search <index_file> <query_file.fvecs> [iterations=10] [output_file] <search_width>"
                 " <internal_topk> <team_size>"
              << std::endl;
    return 1;
  }

  std::string mode = argv[1];

  raft::device_resources dev_resources;

  // Pool memory resource with a 3 GiB initial pool; all allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 3 * 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  if (mode == "build") {
    if (argc < 5) {
      std::cerr << "Usage for build mode: " << argv[0] << " build <base_file.fvecs> <index_file> <degree>"
                << std::endl;
      return 1;
    }

    std::string base_file  = argv[2];
    std::string index_file = argv[3];
    int degree             = std::stoi(argv[4]);

    std::cout << "Reading base vectors from: " << base_file << std::endl;
    int64_t n_samples, n_dim;
    std::vector<float> h_base;
    try {
      h_base = read_fvecs(base_file, n_samples, n_dim);
      std::cout << "Base dataset: " << n_samples << " vectors of dimension " << n_dim << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Error reading base file: " << e.what() << std::endl;
      return 1;
    }

    HostMdspanT host_md(h_base.data(), n_samples, n_dim);

    cagra_build_and_save(dev_resources, host_md, index_file, degree);

  } else if (mode == "search") {
    if (argc < 9) {
      std::cerr << "Usage for search mode: " << argv[0]
                << " search <index_file> <query_file.fvecs> [iterations=10] [output_file] <search_width>"
                   " <internal_topk> <team_size>"
                << std::endl;
      return 1;
    }

    std::string index_file = argv[2];
    std::string query_file = argv[3];

    int iterations          = 10;
    std::string output_file = "";

    // argv[4] is the iteration count when it is a number, otherwise the output file
    bool is_number = true;
    for (size_t i = 0; i < strlen(argv[4]); i++) {
      if (!isdigit(argv[4][i])) {
        is_number = false;
        break;
      }
    }

    if (is_number) {
      iterations  = std::stoi(argv[4]);
      output_file = argv[5];
    } else {
      output_file = argv[4];
    }
    int search_width  = std::stoi(argv[6]);
    int internal_topk = std::stoi(argv[7]);
    int team_size     = std::stoi(argv[8]);

    std::cout << "Reading query vectors from: " << query_file << std::endl;
    int64_t n_queries, q_dim;
    std::vector<float> h_queries;

    try {
      h_queries = read_fvecs(query_file, n_queries, q_dim);
      std::cout << "Query dataset: " << n_queries << " vectors of dimension " << q_dim << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "Error reading query file: " << e.what() << std::endl;
      return 1;
    }

    auto queries = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, q_dim);

    raft::update_device(queries.data_handle(), h_queries.data(), n_queries * q_dim,
                        raft::resource::get_cuda_stream(dev_resources));

    raft::resource::sync_stream(dev_resources);

    std::cout << "Query data transferred to GPU, starting CAGRA search..." << std::endl;

    cagra_load_and_search(dev_resources, raft::make_const_mdspan(queries.view()), index_file, iterations,
                          output_file, search_width, internal_topk, team_size);

  } else {
    std::cerr << "Unknown mode: " << mode << std::endl;
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  Build index: " << argv[0] << " build <base_file.fvecs> <index_file> <degree>" << std::endl;
    std::cerr << "  Search index: " << argv[0]
              << " search <index_file> <query_file.fvecs> [iterations=10] [output_file] <search_width>"
                 " <internal_topk> <team_size>"
              << std::endl;
    return 1;
  }

  std::cout << "CAGRA processing completed." << std::endl;
  return 0;
}
