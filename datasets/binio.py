"""Read and write the vector file formats used in this repository.

fbin / ibin: int32 n, int32 d, then n * d float32 (fbin) or int32 (ibin) values in row-major order.
fvecs / ivecs: for each row, int32 d followed by d float32 (fvecs) or int32 (ivecs) values.
"""
from pathlib import Path

import numpy as np


def read_bin(path: Path, dtype=np.float32, mmap: bool = False) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    if mmap:
        return np.memmap(path, dtype=dtype, mode='r', offset=8, shape=(int(n), int(d)))
    return np.fromfile(path, dtype=dtype, offset=8).reshape(int(n), int(d))


def write_bin(path: Path, a: np.ndarray, dtype=np.float32) -> None:
    a = np.ascontiguousarray(a, dtype=dtype)
    with open(path, 'wb') as f:
        np.array(a.shape, dtype=np.int32).tofile(f)
        a.tofile(f)


def read_vecs(path: Path, dtype=np.float32) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy().view(dtype)


def write_vecs(path: Path, a: np.ndarray, dtype=np.float32) -> None:
    a = np.ascontiguousarray(a, dtype=dtype)
    n, d = a.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = a.view(np.int32)
    out.tofile(path)
