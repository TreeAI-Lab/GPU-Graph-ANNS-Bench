"""Faiss HNSW build driver (Faiss 1.15.1, IndexHNSWFlat). HNSW_BUILD_SECONDS is the time of index.add.
Usage: python hnsw_build.py <base.fbin> <out.faiss> <M> <efConstruction>
"""
import argparse
import time

import faiss
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("M", type=int)
    ap.add_argument("efConstruction", type=int)
    a = ap.parse_args()
    n, d = np.fromfile(a.base, dtype=np.int32, count=2)
    x = np.fromfile(a.base, dtype=np.float32, offset=8).reshape(int(n), int(d))
    idx = faiss.IndexHNSWFlat(int(d), a.M)
    idx.hnsw.efConstruction = a.efConstruction
    t0 = time.time()
    idx.add(x)
    print(f"HNSW_BUILD_SECONDS {time.time() - t0:.3f}")
    faiss.write_index(idx, a.out)


if __name__ == "__main__":
    main()
