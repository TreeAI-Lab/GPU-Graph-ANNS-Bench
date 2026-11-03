"""Write the first k columns of an ibin ground-truth file (int32 n, int32 k, then n x k int32 ids).

The search programs compute Recall@10 against the top-10 ground truth; GANNS requires a file with exactly 10 columns.
usage: python make_gt_k10.py <gt.ibin> <out.ibin> [--k 10]
"""
import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('gt', type=Path, help='ground truth with at least k columns (gt.ibin, k = 100)')
    ap.add_argument('out', type=Path, help='output ibin with k columns')
    ap.add_argument('--k', type=int, default=10)
    args = ap.parse_args()
    n, k = (int(x) for x in np.fromfile(args.gt, dtype=np.int32, count=2))
    if k < args.k:
        raise SystemExit(f'{args.gt}: only {k} columns')
    ids = np.fromfile(args.gt, dtype=np.int32, offset=8).reshape(n, k)[:, :args.k]
    tmp = args.out.with_suffix(args.out.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        np.array([n, args.k], dtype=np.int32).tofile(f)
        np.ascontiguousarray(ids).tofile(f)
    tmp.replace(args.out)
    print(f'wrote {args.out}: {n} x {args.k}')


if __name__ == '__main__':
    main()
