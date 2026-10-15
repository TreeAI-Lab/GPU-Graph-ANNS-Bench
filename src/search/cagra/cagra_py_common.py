"""Helpers for the CAGRA search drivers (cuVS 25.02 Python API).

cuVS 25.02 Python has no Index.update_dataset(), so a graph-only index (saved with include_dataset=False) cannot be
searched. ensure_index_with_dataset() loads the index if it includes the dataset; otherwise it builds the index
once more with the same IndexParams from the base vectors and caches it with the dataset.
"""
from __future__ import annotations

import ctypes
import time
from pathlib import Path

import numpy as np
from cuda import cudart
from pylibraft.common import device_ndarray
from cuvs.neighbors import cagra


def _cuda_check(err):
    if isinstance(err, tuple):
        err, *rest = err
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"CUDA error: {err}")
        if len(rest) == 1:
            return rest[0]
        return rest
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return None


def pinned_array(shape, dtype):
    """numpy array backed by cudaMallocHost memory. Returns (array, host_ptr)."""
    dt = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dt.itemsize
    ptr = _cuda_check(cudart.cudaMallocHost(nbytes))
    buf = (ctypes.c_char * nbytes).from_address(int(ptr))
    arr = np.frombuffer(buf, dtype=dt).reshape(shape)
    return arr, ptr


def read_fbin_header(path: str):
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return int(n), int(d)


def read_fbin(path: str) -> np.ndarray:
    n, d = read_fbin_header(path)
    return np.fromfile(path, dtype=np.float32, offset=8).reshape(n, d)


def read_ibin(path: str) -> np.ndarray:
    n, k = np.fromfile(path, dtype=np.int32, count=2)
    return np.fromfile(path, dtype=np.int32, offset=8).reshape(int(n), int(k))


def recall_at_k(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred, gt: (nq, k) int"""
    nq, k = pred.shape
    hits = 0
    for i in range(nq):
        hits += len(set(pred[i].tolist()) & set(gt[i].tolist()))
    return hits / float(nq * k)


def host_rss_mib() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def gpu_mem_mib():
    err, free_b, tot_b = cudart.cudaMemGetInfo()
    _cuda_check(err)
    used = (tot_b - free_b) / 1048576.0
    return used, free_b / 1048576.0, tot_b / 1048576.0


def append_mem(path: Path, stage: str):
    used, free, tot = gpu_mem_mib()
    rss = host_rss_mib()
    new = not path.exists()
    with path.open("a") as f:
        if new:
            f.write("stage\tgpu_used_mib\tgpu_free_mib\tgpu_total_mib\thost_rss_mib\n")
        f.write(f"{stage}\t{used:.1f}\t{free:.1f}\t{tot:.1f}\t{rss:.1f}\n")


def ensure_index_with_dataset(
    index_path: str,
    base_path: str,
    cache_path: Path,
    graph_degree: int,
    intermediate_graph_degree: int,
    nn_descent_niter: int,
) -> cagra.Index:
    idx = cagra.load(index_path)
    if idx.dim and idx.dim > 0:
        print(f"[cagra] loaded index with dataset dim={idx.dim} from {index_path}", flush=True)
        return idx

    if cache_path.is_file():
        idx2 = cagra.load(str(cache_path))
        if idx2.dim and idx2.dim > 0:
            print(f"[cagra] loaded cached index_with_dataset from {cache_path}", flush=True)
            return idx2

    print(
        "[cagra] WARNING: graph-only index (Python cuvs 25.02 has no update_dataset). "
        f"Rebuilding once with GD={graph_degree} IGD={intermediate_graph_degree} "
        f"niter={nn_descent_niter} and caching to {cache_path}",
        flush=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    base = read_fbin(base_path)
    # CUDA initialization before the timer, as in cagra_build.py
    _ = device_ndarray(np.zeros(1, dtype=np.float32))
    t0 = time.perf_counter()
    xd = device_ndarray(base)
    params = cagra.IndexParams(
        metric="sqeuclidean",
        graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
        build_algo="nn_descent",
        nn_descent_niter=nn_descent_niter,
    )
    idx_new = cagra.build(params, xd)
    cagra.save(str(cache_path), idx_new, include_dataset=True)
    print(f"[cagra] rebuild+cache took {time.perf_counter()-t0:.2f}s -> {cache_path}", flush=True)
    # Reload from disk instead of searching the index object returned by build, which can fault on its first search.
    del idx_new, xd, base
    return cagra.load(str(cache_path))
