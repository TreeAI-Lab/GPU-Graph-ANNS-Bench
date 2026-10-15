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

#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <queue>
#include <sys/stat.h>    // fstat, stat, struct stat
#include <sys/mman.h>    // mmap, munmap, PROT_READ, MAP_SHARED, MAP_FAILED
#include <fcntl.h>       // open
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
#include <cmath>
#include <cstdint>

inline uint32_t log2ceil(uint32_t x)
{
    return static_cast<uint32_t>(std::ceil(std::log2(static_cast<double>(x))));
}

void save_cagra_graph_as_binary(
    const cuvs::neighbors::cagra::index<float, uint32_t>& index,
    raft::resources const& res,
    const std::string& path)
{
    auto graph = index.graph();
    auto n = graph.extent(0);
    auto degree = graph.extent(1);
    uint32_t offset = log2ceil(degree);
    printf("Graph has %zu nodes with degree %zu\n", n, degree);

    std::vector<uint32_t> h_graph(n * degree);
    raft::update_host(h_graph.data(), graph.data_handle(), h_graph.size(), raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) throw std::runtime_error("Cannot open output file");

    // Header: number of nodes and log2 of the degree, the layout make_graph_to_cagra reads
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    ofs.write(reinterpret_cast<const char*>(&offset), sizeof(uint32_t));
    // Adjacency list, n * degree uint32 ids
    ofs.write(reinterpret_cast<const char*>(h_graph.data()), h_graph.size() * sizeof(uint32_t));
    ofs.close();

    std::cout << "Saved binary adjacency list to " << path << std::endl;
}


void extract_graph_from_file(const std::string& in_path, const std::string& out_path) {
    raft::resources res;
    // correct template order: value type then index type
    cuvs::neighbors::cagra::index<float, uint32_t> index(res);

    std::ifstream is(in_path, std::ios::binary);
    if (!is.is_open()) throw std::runtime_error("Cannot open " + in_path);

    // call deserialize with istream & and pointer to index
    cuvs::neighbors::cagra::deserialize(res, is, &index);
    save_cagra_graph_as_binary(index, res, out_path);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <cagra_index_file> <output_graph.bin>\n";
        return 1;
    }
    extract_graph_from_file(argv[1], argv[2]);
    return 0;
}
