# Environments

All measurements ran on Ubuntu 22.04 with an Intel Xeon Gold 6348 CPU and one NVIDIA A800 80GB PCIe GPU, except the
hardware-counter experiments (Tables 9 and 10, Figure 14), which ran on an NVIDIA A10 24GB in a virtual machine with
GPU passthrough, because reading performance counters needs administrator rights on the GPU.

Five conda environments are created under `$W/envs` (see `env.sh`):

| Environment | Purpose | CUDA | Compilers |
|---|---|---|---|
| `build` | SONG, GGNN, GANNS, DiskANN, BANG, the drivers in `src/` | 12.4 | gcc 12 |
| `cagra` | CAGRA graph construction with the cuVS 25.02.01 packages | 12.4 | |
| `pilot` | PilotANN, Faiss HNSW on the CPU (faiss-cpu 1.15.1), PyTorch 2.5.1 | 12.4 | gcc 12 |
| `faiss-build` | Faiss with GPU support built from source (IVF-PQ) | 12.6 | gcc 11 |
| `cuvs-build` | cuVS 25.02 built from source (CAGRA search driver, IVF-PQ, brute force) | 12.9 | gcc 14 |

```bash
source env.sh
bash environment/create_envs.sh            # from the .yml files
bash environment/create_envs.sh --locked   # the exact package builds used for the paper (locks/*.txt)
```

`create_envs.sh` installs PyTorch 2.5.1 (CUDA 12.4 wheels) into `pilot` with pip.

The locked `faiss-build` environment lacks `libcurand-dev` and `cuda-profiler-api`, which the Faiss GPU build needs;
`faiss-build.yml` includes them. Install them into the locked environment before building Faiss.
