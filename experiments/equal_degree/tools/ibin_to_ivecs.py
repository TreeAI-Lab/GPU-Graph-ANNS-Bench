"""Convert a ground-truth file from ibin (int32 n, int32 d, then n x d ids) to ivecs (every row prefixed by d).

GANNS and the PilotANN loader read the ground truth only as ivecs.

Usage: python3 ibin_to_ivecs.py <gt.ibin> <groundtruth.ivecs>
"""
import argparse

import numpy as np


def ibin_to_ivecs(src: str, dst: str) -> None:
    a = np.fromfile(src, dtype='int32')
    n, d = int(a[0]), int(a[1])
    ids = a[2:2 + n * d].reshape(n, d)
    out = np.empty((n, d + 1), dtype='int32')
    out[:, 0] = d
    out[:, 1:] = ids
    out.tofile(dst)
    print(f'{dst}: {n} rows of {d} ids')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('src', help='ground truth in ibin format')
    parser.add_argument('dst', help='output ivecs file')
    args = parser.parse_args()
    ibin_to_ivecs(args.src, args.dst)


if __name__ == '__main__':
    main()
