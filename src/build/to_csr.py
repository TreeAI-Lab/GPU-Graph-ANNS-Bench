"""Convert the native graph file of a builder into the CSR file (graph.csr) read by unified_search.

graph.csr: int64 n, int64 nnz, int64 offsets[n+1], int32 nbrs[nnz].
Only neighbors in [0, n) that are not self loops are kept, in their original order. Duplicate edges are kept
(unified_search stats counts them separately, the search skips them through its visited set).

Usage: python src/build/to_csr.py <format> <input> <out.csr> [--n N] [--kbuild K] [--shift S]
format: song, ganns, ggnn, cagra, faiss_hnsw, faiss_nsg, diskann, npz, pilotann_pt, bang_disk
"""
import argparse
import io
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def write_csr(path: Path, lists: list[np.ndarray]) -> None:
    n = len(lists)
    deg = np.fromiter((len(a) for a in lists), dtype=np.int64, count=n)
    off = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=off[1:])
    adj = np.concatenate(lists).astype(np.int32) if n else np.zeros(0, np.int32)
    with open(path, "wb") as f:
        np.array([n, adj.size], dtype=np.int64).tofile(f)
        off.tofile(f)
        adj.tofile(f)
    logger.info("wrote %s: n=%d nnz=%d avg_deg=%.2f", path, n, adj.size, adj.size / max(n, 1))


def clean(row: np.ndarray, u: int, n: int) -> np.ndarray:
    return row[(row >= 0) & (row < n) & (row != u)]


def from_padded(mat: np.ndarray) -> list[np.ndarray]:
    n = mat.shape[0]
    return [clean(mat[u], u, n) for u in range(n)]


def load_song(path: Path, n: int, shift: int) -> list[np.ndarray]:
    """SONG bfsg.graph: n rows of 2^shift uint64, the first holding the number of neighbors."""
    stride = 1 << shift
    mat = np.fromfile(path, dtype=np.uint64, count=n * stride).reshape(n, stride).astype(np.int64)
    return [clean(mat[u, 1 : 1 + int(mat[u, 0])], u, n) for u in range(n)]


def load_ganns(path: Path) -> list[np.ndarray]:
    """GANNS graph: int32 n, int32 log2 of the row length, then the padded rows."""
    n, s_bits = np.fromfile(path, dtype=np.int32, count=2)
    mat = np.fromfile(path, dtype=np.int32, offset=8, count=int(n) << int(s_bits)).reshape(int(n), 1 << int(s_bits))
    return from_padded(mat)


def load_ggnn(path: Path, n: int, kbuild: int) -> list[np.ndarray]:
    # The first N x KBuild int32 of the GPU memory dump are the bottom layer (graph_config.cpp: Ns_offsets[0] = 0)
    mat = np.fromfile(path, dtype=np.int32, count=n * kbuild).reshape(n, kbuild)
    return from_padded(mat)


def load_cagra(path: Path) -> list[np.ndarray]:
    """CAGRA file written by cagra.save: consecutive .npy blocks; the first 2-D block is the graph (n, degree)."""
    from numpy.lib import format as npf
    raw = Path(path).read_bytes()
    f = io.BytesIO(raw)
    while True:
        p = raw.find(b"\x93NUMPY", f.tell())
        if p < 0:
            raise ValueError("no 2-D npy block found in CAGRA file")
        f.seek(p)
        ver = npf.read_magic(f)
        reader = npf.read_array_header_1_0 if ver[0] == 1 else npf.read_array_header_2_0
        shape, _, dtype = reader(f)
        start = f.tell()
        if len(shape) == 2:
            mat = np.frombuffer(raw, dtype=dtype, count=shape[0] * shape[1], offset=start).reshape(shape)
            return from_padded(mat.astype(np.int64))
        f.seek(start + (int(np.prod(shape)) if shape else 1) * dtype.itemsize)


def load_faiss_hnsw(path: Path) -> list[np.ndarray]:
    """Bottom layer of a Faiss IndexHNSWFlat file, parsed directly (Faiss v1.15.1 impl/index_write.cpp:357-373 and
    868-904).

    Reading hnsw.neighbors (a MaybeOwnedVector) through the Faiss 1.15 Python bindings crashes, so the bindings are
    not used. Layout: fourcc "IHNf", index header d(i32) ntotal(i64) dummy(i64) dummy(i64) is_trained(u8) metric(i32)
    [metric_arg(f32)], then five vectors, each a size_t length followed by the data: assign_probas(f64)
    cum_nneighbor_per_level(i32) levels(i32) offsets(u64) neighbors(i32).
    """
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    if bytes(raw[:4]) != b"IHNf":
        raise ValueError(f"not an IndexHNSWFlat file: {bytes(raw[:4])!r}")
    pos = 4
    (d,) = struct.unpack_from("<i", raw, pos); pos += 4
    (ntotal,) = struct.unpack_from("<q", raw, pos); pos += 8 + 8 + 8 + 1
    (metric,) = struct.unpack_from("<i", raw, pos); pos += 4
    if metric > 1:
        pos += 4

    def vec(dtype):
        nonlocal pos
        (size,) = struct.unpack_from("<Q", raw, pos); pos += 8
        a = np.frombuffer(raw, dtype=dtype, count=size, offset=pos); pos += size * np.dtype(dtype).itemsize
        return a

    vec(np.float64)
    cum = vec(np.int32).astype(np.int64)
    levels = vec(np.int32)
    off = vec(np.uint64).astype(np.int64)
    nb = vec(np.int32).astype(np.int64)
    n = int(ntotal)
    if levels.size != n or off.size != n + 1:
        raise ValueError("inconsistent HNSW sizes")
    return [clean(nb[off[u] + cum[0] : off[u] + cum[1]], u, n) for u in range(n)]


def load_faiss_nsg(path: Path) -> list[np.ndarray]:
    import faiss
    idx = faiss.downcast_index(faiss.read_index(str(path)))
    g = idx.nsg.get_final_graph()
    n, k = g.N, g.K
    mat = faiss.rev_swig_ptr(g.data, n * k).reshape(n, k).astype(np.int64)
    return from_padded(mat)


def load_diskann(path: Path) -> list[np.ndarray]:
    """DiskANN in-memory index: u64 file size, u32 max degree, u32 start, u64 frozen points, then per node
    u32 k and k u32 neighbors. Frozen points at the end are dropped."""
    raw = Path(path).read_bytes()
    size, _max_deg, _start, frozen = struct.unpack_from("<QIIQ", raw, 0)
    if size != len(raw):
        raise ValueError(f"DiskANN header says {size} bytes, file has {len(raw)}")
    lists, pos = [], 24
    while pos < len(raw):
        (k,) = struct.unpack_from("<I", raw, pos)
        lists.append(np.frombuffer(raw, dtype=np.uint32, count=k, offset=pos + 4).astype(np.int64))
        pos += 4 + 4 * k
    n = len(lists) - int(frozen)
    return [clean(lists[u], u, n) for u in range(n)]


def load_bang_disk(path: Path, dim=None, max_deg=None) -> list[np.ndarray]:
    """BANG / DiskANN *_disk.bin: per-node record = float32[dim] + uint32 nnbr + uint32[max_deg].

    If dim or max_deg is not given, it is read from the *_disk_metadata.bin next to the file when present, else
    taken as 96 and 64 (DEEP1M with degree 64).
    """
    path = Path(path)
    meta = path.with_name(path.name.replace("_disk.bin", "_disk_metadata.bin"))
    if (dim is None or max_deg is None) and meta.is_file() and meta.stat().st_size >= 32:
        # metadata as uint32: dim at word 0, max_deg at word 6
        u32 = np.fromfile(meta, dtype=np.uint32)
        if dim is None:
            dim = int(u32[0])
        if max_deg is None:
            max_deg = int(u32[6])
    if dim is None:
        dim = 96
    if max_deg is None:
        max_deg = 64
    rec = dim * 4 + 4 + max_deg * 4
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    if raw.size % rec != 0:
        raise ValueError(
            f"bang disk size {raw.size} not divisible by record {rec} (dim={dim} max_deg={max_deg})"
        )
    n = int(raw.size // rec)
    # strided views over the neighbor counts and the padded neighbor slots
    ks = np.ndarray(n, dtype=np.uint32, buffer=raw, offset=dim * 4, strides=(rec,))
    nbr_mat = np.ndarray(
        (n, max_deg), dtype=np.uint32, buffer=raw, offset=dim * 4 + 4, strides=(rec, 4)
    )
    if ks.max() > max_deg:
        raise ValueError(f"nnbr max {ks.max()} > max_deg {max_deg}")
    lists: list[np.ndarray] = [None] * n  # type: ignore
    for u in range(n):
        lists[u] = clean(nbr_mat[u, : int(ks[u])].astype(np.int64, copy=False), u, n)
    return lists


def load_pilotann_pt(path: Path, graph: str = "fullgraph") -> list[np.ndarray]:
    """PilotANN torch checkpoint: state_dict['{graph}.indptr'|'{graph}.indices']."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    key_ptr = f"{graph}.indptr"
    key_idx = f"{graph}.indices"
    if key_ptr not in sd or key_idx not in sd:
        raise KeyError(f"missing {key_ptr}/{key_idx}; have {list(sd)}")
    indptr = sd[key_ptr].cpu().numpy().astype(np.int64)
    indices = sd[key_idx].cpu().numpy().astype(np.int64)
    n = indptr.size - 1
    return [clean(indices[indptr[u] : indptr[u + 1]], u, n) for u in range(n)]


def load_npz(path: Path) -> list[np.ndarray]:
    """npz with indptr and indices (the PilotANN full graph written by pilot_build.py)."""
    z = np.load(path)
    indptr, indices = z["indptr"].astype(np.int64), z["indices"].astype(np.int64)
    n = indptr.size - 1
    return [clean(indices[indptr[u] : indptr[u + 1]], u, n) for u in range(n)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "format",
        choices=[
            "song",
            "ganns",
            "ggnn",
            "cagra",
            "faiss_hnsw",
            "faiss_nsg",
            "diskann",
            "npz",
            "pilotann_pt",
            "bang_disk",
        ],
    )
    ap.add_argument("input", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--kbuild", type=int, default=None)
    ap.add_argument("--shift", type=int, default=None)
    ap.add_argument("--graph", type=str, default="fullgraph", help="pilotann_pt: fullgraph|subgraph")
    ap.add_argument("--dim", type=int, default=None, help="bang_disk vector dim")
    ap.add_argument("--max-deg", type=int, default=None, help="bang_disk max degree pad")
    a = ap.parse_args()
    loaders = {
        "song": lambda: load_song(a.input, a.n, a.shift),
        "ganns": lambda: load_ganns(a.input),
        "ggnn": lambda: load_ggnn(a.input, a.n, a.kbuild),
        "cagra": lambda: load_cagra(a.input),
        "faiss_hnsw": lambda: load_faiss_hnsw(a.input),
        "faiss_nsg": lambda: load_faiss_nsg(a.input),
        "diskann": lambda: load_diskann(a.input),
        "npz": lambda: load_npz(a.input),
        "pilotann_pt": lambda: load_pilotann_pt(a.input, a.graph),
        "bang_disk": lambda: load_bang_disk(a.input, a.dim, a.max_deg),
    }
    write_csr(a.out, loaders[a.format]())


if __name__ == "__main__":
    main()
