// Attaches base vectors (fvecs) to a graph-only CAGRA index and writes the index with the dataset.
// Usage: attach_dataset <index_in> <dataset.fvecs> <index_out>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <queue>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unordered_set>
#include <cuda_fp16.h>
#include <cassert>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/serialize.hpp>

#include <cuvs/neighbors/cagra.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

using namespace std;

// simple fvecs reader -> returns raft::host_matrix constructed via raft::make_host_matrix
static raft::host_matrix<float, int64_t> read_fvecs(const raft::resources &res, const string &path) {
    ifstream in(path, ios::binary);
    if (!in) throw runtime_error("cannot open fvecs: " + path);

    in.seekg(0, ios::end);
    size_t file_size = in.tellg();
    in.seekg(0, ios::beg);

    vector<char> buf(file_size);
    in.read(buf.data(), file_size);
    in.close();

    size_t offset = 0;
    if (file_size < sizeof(int)) throw runtime_error("invalid fvecs file");
    int dim = *reinterpret_cast<int*>(buf.data());
    size_t entry_size = sizeof(int) + sizeof(float) * dim;
    if (entry_size == 0) throw runtime_error("invalid fvecs dim");
    int64_t n = static_cast<int64_t>(file_size / entry_size);

    // allocate host matrix using RAFT helper
    auto host = raft::make_host_matrix<float, int64_t>(n, static_cast<int64_t>(dim));

    for (int64_t i = 0; i < n; ++i) {
        int d = *reinterpret_cast<int*>(buf.data() + offset);
        if (d != dim) throw runtime_error("inconsistent dim in fvecs");
        offset += sizeof(int);
        // copy row into contiguous host matrix storage
        float* dst = host.data_handle() + i * dim;
        memcpy(dst, buf.data() + offset, sizeof(float) * dim);
        offset += sizeof(float) * dim;
    }
    return host;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: attach_dataset <index_in> <dataset_fvecs> <index_out>\n";
        return 1;
    }
    string index_in = argv[1];
    string fvecs = argv[2];
    string index_out = argv[3];

    try {
        // create RAFT resources
        raft::resources res;

        // load existing index (type params must match how index was created)
        cuvs::neighbors::cagra::index<float, uint32_t> index(res);

        {
            ifstream is(index_in, ios::binary);
            if (!is) throw runtime_error("cannot open index file: " + index_in);
            cuvs::neighbors::cagra::deserialize(res, is, &index);
            is.close();
        }

        // read fvecs and update dataset
        auto host_dataset = read_fvecs(res, fvecs);
        auto host_view = raft::make_const_mdspan(host_dataset.view());
        index.update_dataset(res, host_view);

        // serialize index with dataset included
        {
            ofstream os(index_out, ios::binary);
            if (!os) throw runtime_error("cannot open output file: " + index_out);
            cuvs::neighbors::cagra::serialize(res, os, index);
            os.close();
        }

        cout << "Dataset attached and saved to: " << index_out << endl;
    } catch (const std::exception &e) {
        cerr << "ERROR: " << e.what() << endl;
        return 2;
    }
    return 0;
}