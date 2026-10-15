// GGNN build and query driver of the original manuscript, linked against the libraries of the patched GGNN tree
// (ggnn-orig), whose query() also returns the counters and stage clocks. Built by ../../build_ggnn.sh.
//
//   ggnn_main_gpu_data build <base.fvecs> <graph_dir> <KBuild> <tau_build>
//   ggnn_main_gpu_data query <base.fvecs> <query.fvecs> <graph_dir> <KBuild> <KQuery> <max_iterations> <l2|cos>
//                            <result_file>
//
// query prints the search time of all queries, the number of distance calculations and of visited nodes
// (path count), and the shares of the stage clocks: fetching and filtering neighbors, distance calculation,
// and the update of the candidate list.

#include <ggnn/base/ggnn.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <curand.h>

float* read_fvecs(const std::string& filename, size_t& num_vectors, size_t& dim)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Unable to open .fvecs file");
    }
    int dim_int;
    input.read(reinterpret_cast<char*>(&dim_int), sizeof(int));
    dim = static_cast<size_t>(dim_int);
    input.seekg(0, std::ios::end);
    size_t filesize = input.tellg();
    input.seekg(0, std::ios::beg);
    size_t record_size = (dim + 1) * sizeof(float);
    num_vectors = filesize / record_size;
    float* data = new float[num_vectors * dim];
    for (size_t i = 0; i < num_vectors; ++i) {
        int d;
        input.read(reinterpret_cast<char*>(&d), sizeof(int));
        if (static_cast<size_t>(d) != dim) {
            throw std::runtime_error("Inconsistent vector dimension");
        }
        input.read(reinterpret_cast<char*>(data + i * dim), sizeof(float) * dim);
    }
    return data;
}

using namespace ggnn;

int main(int argc, char** argv)
{
    using GGNN = ggnn::GGNN<int32_t, float>;
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mode (build|query)> [other arguments]" << std::endl;
        std::cerr << "Build mode: " << argv[0] << " build <base.fvecs> <graph_dir> <KBuild> <tau_build>" << std::endl;
        std::cerr << "Query mode: " << argv[0]
                  << " query <base.fvecs> <query.fvecs> <graph_dir> <KBuild> <KQuery> <max_iterations>"
                     " <distance_measure (l2|cos)> <result_file>"
                  << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "build") {
        if (argc != 6) {
            std::cerr << "Usage for build: " << argv[0] << " build <base.fvecs> <graph_dir> <KBuild> <tau_build>"
                      << std::endl;
            return 1;
        }
        std::string base_path = argv[2];
        std::string graph_dir = argv[3];
        const uint32_t KBuild = std::stoi(argv[4]);
        const float tau_build = std::stof(argv[5]);
        DistanceMeasure measure = DistanceMeasure::Euclidean;
        size_t N_base, D;
        float* h_base = read_fvecs(base_path, N_base, D);
        float* base;
        cudaMalloc(&base, N_base * D * sizeof(float));
        cudaMemcpy(base, h_base, N_base * D * sizeof(float), cudaMemcpyHostToDevice);
        GGNN ggnn{};
        int32_t gpu_id = 0;

        ggnn.setBase(ggnn::Dataset<float>::referenceGPUData(base, N_base, D, gpu_id));
        auto start = std::chrono::high_resolution_clock::now();
        // 3 refinement iterations
        ggnn.build(KBuild, tau_build, 3, measure);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> build_duration = end - start;
        std::cout << "Graph built in " << build_duration.count() << " ms." << std::endl;
        ggnn.setWorkingDirectory(graph_dir);
        ggnn.store();
        std::cout << "Graph built and stored successfully." << std::endl;
        cudaFree(base);
        delete[] h_base;
    }
    else if (mode == "query") {
        if (argc != 10) {
            std::cerr << "Usage for query: " << argv[0]
                      << " query <base.fvecs> <query.fvecs> <graph_dir> <KBuild> <KQuery> <max_iterations>"
                         " <distance_measure (l2|cos)> <result_file>"
                      << std::endl;
            return 1;
        }
        std::string base_path = argv[2];
        std::string query_path = argv[3];
        std::string graph_dir = argv[4];
        const uint32_t KBuild = std::stoi(argv[5]);
        const uint32_t KQuery = std::stoi(argv[6]);
        const uint32_t max_iterations = std::stoi(argv[7]);
        std::string dist_type = argv[8];
        std::string result_filename = argv[9];
        DistanceMeasure measure;
        if (dist_type == "l2") {
            measure = DistanceMeasure::Euclidean;
        } else if (dist_type == "cos") {
            measure = DistanceMeasure::Cosine;
        } else {
            std::cerr << "Invalid distance measure. Use 'l2' or 'cos'." << std::endl;
            return 1;
        }
        size_t N_base, D;
        float* h_base = read_fvecs(base_path, N_base, D);
        float* base;
        cudaMalloc(&base, N_base * D * sizeof(float));

        GGNN ggnn{};
        int32_t gpu_id = 0;
        ggnn.setWorkingDirectory(graph_dir);
        ggnn.setGPUs({0});
        ggnn.setBase(ggnn::Dataset<float>::referenceGPUData(base, N_base, D, gpu_id));
        size_t N_query;
        float* h_query = read_fvecs(query_path, N_query, D);
        float* query;
        cudaMalloc(&query, N_query * D * sizeof(float));
        auto data_start = std::chrono::high_resolution_clock::now();
        cudaMemcpy(base, h_base, N_base * D * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(query, h_query, N_query * D * sizeof(float), cudaMemcpyHostToDevice);
        auto data_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> data_duration = data_end - data_start;
        std::cout << "Data transfer to GPU completed in " << data_duration.count() / 1000.0 << " ms." << std::endl;
        ggnn::Dataset<float> d_query = ggnn::Dataset<float>::referenceGPUData(query, N_query, D, gpu_id);

        printf("Loading graph...\n");
        ggnn.load(KBuild);

        unsigned long long h_calc_count = 0;
        unsigned long long h_path_count = 0;
        unsigned long long h_fetch_time = 0;
        unsigned long long h_filter_time = 0;
        unsigned long long h_calc_time = 0;
        unsigned long long h_update_time = 0;

        printf("Start query with KBuild=%u, KQuery=%u, max_iterations=%u\n", KBuild, KQuery, max_iterations);

        // tau_query = 0.5
        auto query_start = std::chrono::high_resolution_clock::now();
        const auto [indices, dists] = ggnn.query(d_query, KQuery, 0.5f, h_calc_count, h_path_count, h_fetch_time,
                                                 h_filter_time, h_calc_time, h_update_time, max_iterations, measure);
        cudaDeviceSynchronize();
        auto query_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> query_duration = query_end - query_start;
        std::cout << "Query completed in " << query_duration.count() << " us." << std::endl;
        std::cout << "calc count: " << h_calc_count << std::endl;
        std::cout << "path count: " << h_path_count << std::endl;
        unsigned long long sum = h_fetch_time + h_filter_time + h_calc_time + h_update_time;
        std::cout << "fetch time ratio: " << (h_fetch_time + h_filter_time) / (double)sum << std::endl;
        std::cout << "calc time ratio: " << h_calc_time / (double)sum << std::endl;
        std::cout << "update time ratio: " << h_update_time / (double)sum << std::endl;

        std::ofstream result_file(result_filename);
        if (!result_file.is_open()) {
            std::cerr << "Failed to open output file!" << std::endl;
            return -1;
        }
        for (size_t i = 0; i < N_query; ++i) {
            for (int j = 0; j < KQuery; ++j) {
                result_file << indices[i * KQuery + j];
                if (j < KQuery - 1)
                    result_file << " ";
            }
            result_file << "\n";
        }
        result_file.close();
        cudaFree(query);
        delete[] h_query;
        cudaFree(base);
        delete[] h_base;
    }
    else {
        std::cerr << "Invalid mode. Use 'build' or 'query'." << std::endl;
        return 1;
    }
    return 0;
}
