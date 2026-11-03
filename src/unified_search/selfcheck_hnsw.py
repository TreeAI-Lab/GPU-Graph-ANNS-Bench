"""Check of the common searcher against a library search: builds a Faiss HNSW index (default M 32,
efConstruction 40), reports Recall@10 and distance computations per query of the Faiss search at several
efSearch values, and stores the index so that its bottom layer can be converted to CSR and searched with
unified_search at L = efSearch.

Usage: python src/unified_search/selfcheck_hnsw.py <dataset_dir> <out_index> [--M 32] [--efc 40]
"""
import argparse
import logging
import time
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def read_bin(path: Path, dtype) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.fromfile(path, dtype=dtype, offset=8).reshape(int(n), int(d))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efc", type=int, default=40)
    a = ap.parse_args()
    base, query, gt = read_bin(a.dir / "base.fbin", np.float32), read_bin(a.dir / "query.fbin", np.float32), read_bin(a.dir / "gt.ibin", np.int32)
    idx = faiss.IndexHNSWFlat(base.shape[1], a.M)
    idx.hnsw.efConstruction = a.efc
    t = time.time(); idx.add(base); logger.info("built M=%d efC=%d in %.1fs", a.M, a.efc, time.time() - t)
    faiss.write_index(idx, str(a.out))
    print("efSearch\trecall_at_10\tfaiss_ndis_per_query")
    for ef in [16, 32, 64, 128, 256]:
        idx.hnsw.efSearch = ef
        faiss.cvar.hnsw_stats.reset()
        _, I = idx.search(query, 10)
        rec = np.mean([len(set(I[i]) & set(gt[i, :10])) / 10 for i in range(len(query))])
        print(f"{ef}\t{rec:.5f}\t{faiss.cvar.hnsw_stats.ndis / len(query):.1f}")


if __name__ == "__main__":
    main()
