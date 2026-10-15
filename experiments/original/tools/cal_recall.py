"""Recall@K of a result file against the ground truth, written to a one-line file.

The result file holds one line of neighbor ids per query. Lines before and after them are skipped by count,
as the three recall scripts of the original release did:
  SONG log (stderr and stdout in one file): --skip-head 7 --skip-tail 2
  CAGRA result file of my_cagra (three '#' header lines): --skip-head 3
  GGNN result file: no lines skipped

Recall@K = mean over queries of |top-K ground truth ids  intersect  first K result ids| / K.

With --append the recall line is added to --out, so that the file holds one line per timed run.

Usage: python3 cal_recall.py --gt gt.ibin --ret result.txt --k 10 --out recall.txt [--skip-head N] [--skip-tail N]
       [--append]
"""
import argparse

import numpy as np


def read_gt(filename: str) -> np.ndarray:
    """Ground-truth ids from ivecs (every row prefixed by its length) or ibin (int32 n, int32 d header)."""
    a = np.fromfile(filename, dtype='int32')
    if filename.endswith('.ibin'):
        n, d = int(a[0]), int(a[1])
        return a[2:2 + n * d].reshape(n, d)
    dim = a[0]
    return a.reshape(-1, dim + 1)[:, 1:]


def read_retrieved_txt(filename: str, skip_head: int = 0, skip_tail: int = 0) -> np.ndarray:
    with open(filename, 'r') as f:
        lines = f.readlines()
    valid_lines = lines[skip_head: len(lines) - skip_tail]
    return np.array([[int(x) for x in line.strip().split()] for line in valid_lines])


def compute_recall_at_k(gt: np.ndarray, ret: np.ndarray, k: int = 10) -> float:
    total_recall = 0
    for g, r in zip(gt, ret):
        hit = len(np.intersect1d(g, r[:k]))
        total_recall += hit / len(g)
    return total_recall / len(gt)


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute Recall@K and save to file')
    parser.add_argument('--gt', type=str, required=True, help='ground truth (.ibin or .ivecs)')
    parser.add_argument('--ret', type=str, required=True, help='retrieved result file')
    parser.add_argument('--out', type=str, required=True, help='output file for the recall line')
    parser.add_argument('--k', type=int, default=10, help='Top-K to compute recall at')
    parser.add_argument('--skip-head', type=int, default=0, help='lines to skip at the start of --ret')
    parser.add_argument('--skip-tail', type=int, default=0, help='lines to skip at the end of --ret')
    parser.add_argument('--append', action='store_true', help='append the recall line to --out')
    args = parser.parse_args()

    groundtruth = read_gt(args.gt)[:, :args.k]
    retrieved = read_retrieved_txt(args.ret, args.skip_head, args.skip_tail)

    assert groundtruth.shape[0] == retrieved.shape[0], \
        f'Groundtruth queries: {groundtruth.shape[0]}, Retrieved queries: {retrieved.shape[0]}'

    recall = compute_recall_at_k(groundtruth, retrieved, k=args.k)
    result_line = f'Recall@{args.k} = {recall:.4f}\n'

    print(result_line.strip())

    with open(args.out, 'a' if args.append else 'w') as f:
        f.write(result_line)


if __name__ == '__main__':
    main()
