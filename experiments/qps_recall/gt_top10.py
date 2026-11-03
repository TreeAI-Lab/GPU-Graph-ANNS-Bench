"""Write the first 10 neighbor ids of every query of an ibin ground truth (int32 n, int32 k, then n x k ids).

The search drivers compute Recall@10 against this file. GANNS reads rows of exactly 10 ids, so it needs it.
Usage: python gt_top10.py <gt.ibin> <out.ibin>
"""
import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="ground truth with at least 10 ids per query (gt.ibin, k = 100)")
    ap.add_argument("dst", help="output ibin with 10 ids per query")
    args = ap.parse_args()
    n, k = np.fromfile(args.src, dtype=np.int32, count=2)
    ids = np.fromfile(args.src, dtype=np.int32, offset=8).reshape(int(n), int(k))
    top = np.ascontiguousarray(ids[:, :10])
    with open(args.dst, "wb") as f:
        f.write(np.array([top.shape[0], 10], dtype=np.int32).tobytes())
        f.write(top.tobytes())
    print(f"gt {args.dst} {top.shape}")


if __name__ == "__main__":
    main()
