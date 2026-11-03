"""Write the first K neighbor ids of every query of an ibin ground truth (int32 n, int32 k, then n x k ids).

The GPU top-K sweeps compute Recall@K against this file. GANNS reads rows of exactly K ids, so it needs it.
Usage: python gt_topk.py <gt.ibin> <out.ibin> <K>
"""
import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="ground truth with at least K ids per query (gt.ibin, k = 100)")
    ap.add_argument("dst", help="output ibin with K ids per query")
    ap.add_argument("k", type=int, help="ids per query to keep")
    args = ap.parse_args()
    n, k = np.fromfile(args.src, dtype=np.int32, count=2)
    if args.k > k:
        raise SystemExit(f"{args.src} has {k} ids per query, fewer than {args.k}")
    ids = np.fromfile(args.src, dtype=np.int32, offset=8).reshape(int(n), int(k))
    top = np.ascontiguousarray(ids[:, :args.k])
    with open(args.dst, "wb") as f:
        f.write(np.array([top.shape[0], args.k], dtype=np.int32).tobytes())
        f.write(top.tobytes())
    logger.info("gt %s %s", args.dst, top.shape)


if __name__ == "__main__":
    main()
