// Writes a graph-only CAGRA index from a binary adjacency list:
// uint32 n_nodes, uint32 log2(degree), then n_nodes * 2^log2(degree) uint32 neighbor ids.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <cstring>

struct GraphBinary {
    uint32_t n_nodes;
    uint32_t log_degree;  // log2 of the degree
    std::vector<uint32_t> edges;  // n_nodes * degree
};

GraphBinary load_graph_binary(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open input file: " + path);

    GraphBinary g{};
    ifs.read(reinterpret_cast<char*>(&g.n_nodes), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char*>(&g.log_degree), sizeof(uint32_t));
    uint32_t degree = 1u << g.log_degree;
    g.edges.resize(static_cast<size_t>(g.n_nodes) * degree);
    ifs.read(reinterpret_cast<char*>(g.edges.data()), g.edges.size() * sizeof(uint32_t));

    if (!ifs.good()) throw std::runtime_error("Error reading binary graph file.");
    std::cout << "Loaded graph: " << g.n_nodes << " nodes, degree " << degree << std::endl;
    return g;
}

void save_cagra_index_from_graph(const std::string& in_path, const std::string& out_path) {
    raft::resources res;

    GraphBinary g = load_graph_binary(in_path);
    uint32_t degree = 1u << g.log_degree;
    auto n_nodes = static_cast<int64_t>(g.n_nodes);
    auto deg_i64 = static_cast<int64_t>(degree);

    // build host graph mdarray and copy edges into it (cagra serialize expects host graph)
    auto host_graph = raft::make_host_matrix<uint32_t, int64_t>(n_nodes, deg_i64);
    // ensure contiguous layout, then memcpy
    std::memcpy(host_graph.data_handle(), g.edges.data(), sizeof(uint32_t) * g.edges.size());

    // construct CAGRA index with metric (match deserialize behavior)
    cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
    cuvs::neighbors::cagra::index<float, uint32_t> index(res, metric);

    // update graph (host mdspan)
    index.update_graph(res, raft::make_const_mdspan(host_graph.view()));

    // no dataset included
    bool include_dataset = false;

    // serialize to output file (use serialize overload that accepts ostream)
    std::ofstream os(out_path, std::ios::binary);
    if (!os.is_open()) throw std::runtime_error("Cannot open output: " + out_path);

    cuvs::neighbors::cagra::serialize(res, os, index, include_dataset);
    raft::resource::sync_stream(res);
    os.close();

    std::cout << "Saved CAGRA index to " << out_path << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <graph_binary> <output_cagra_index>\n";
        return 1;
    }
    try {
        save_cagra_index_from_graph(argv[1], argv[2]);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
