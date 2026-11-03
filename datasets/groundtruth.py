"""Exact top-k neighbors (L2) of every query, written to <dataset>/gt.ibin.

Usage: python datasets/groundtruth.py <dataset dir> [--k 100] [--gpu]
--gpu uses Faiss on all visible GPUs; base sets larger than GPU memory are searched in chunks and merged.
"""
import argparse
import logging
from pathlib import Path

import faiss
import numpy as np

from binio import read_bin, write_bin

logger = logging.getLogger(__name__)
CHUNK = 10_000_000  # base vectors per chunk


def search(base: np.ndarray, query: np.ndarray, k: int, gpu: bool) -> np.ndarray:
    dist = np.full((query.shape[0], k), np.inf, dtype=np.float32)
    ids = np.full((query.shape[0], k), -1, dtype=np.int64)
    for start in range(0, base.shape[0], CHUNK):
        chunk = np.ascontiguousarray(base[start:start + CHUNK])
        index = faiss.IndexFlatL2(base.shape[1])
        if gpu:
            index = faiss.index_cpu_to_all_gpus(index)
        index.add(chunk)
        d, i = index.search(query, k)
        dist = np.concatenate([dist, d], axis=1)
        ids = np.concatenate([ids, i + start], axis=1)
        order = np.argsort(dist, axis=1, kind='stable')[:, :k]
        dist, ids = np.take_along_axis(dist, order, 1), np.take_along_axis(ids, order, 1)
        logger.info('searched base [%d, %d)', start, start + chunk.shape[0])
    return ids.astype(np.int32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('dir', type=Path)
    ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--gpu', action='store_true')
    args = ap.parse_args()
    base = read_bin(args.dir / 'base.fbin', mmap=True)
    query = np.ascontiguousarray(read_bin(args.dir / 'query.fbin'))
    gt = search(base, query, args.k, args.gpu)
    write_bin(args.dir / 'gt.ibin', gt, np.int32)
    logger.info('wrote %s %s', args.dir / 'gt.ibin', gt.shape)


if __name__ == '__main__':
    main()
