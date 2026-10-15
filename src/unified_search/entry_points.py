"""Entry points shared by all graphs: the medoid of the dataset (the point closest to the mean) followed by
points drawn at random with a fixed seed.

Writes <dataset_dir>/entry_m<M>.ibin with shape (1, M); the first id is the medoid. The entry points depend only
on the dataset, not on any graph.
Usage: python src/unified_search/entry_points.py <dataset_dir> [--m 8] [--seed 0]
"""
import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    n, d = np.fromfile(a.dir / "base.fbin", dtype=np.int32, count=2)
    x = np.fromfile(a.dir / "base.fbin", dtype=np.float32, offset=8).reshape(int(n), int(d))
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    dist = np.einsum("ij,ij->i", x - mean, x - mean)
    medoid = int(np.argmin(dist))
    rng = np.random.default_rng(a.seed)
    others = [int(i) for i in rng.choice(int(n), size=a.m * 4, replace=False) if i != medoid][: a.m - 1]
    ids = np.array([[medoid] + others], dtype=np.int32)
    out = a.dir / f"entry_m{a.m}.ibin"
    with open(out, "wb") as f:
        np.array(ids.shape, dtype=np.int32).tofile(f)
        ids.tofile(f)
    logger.info("%s: medoid=%d, entries=%s", out, medoid, ids[0].tolist())


if __name__ == "__main__":
    main()
