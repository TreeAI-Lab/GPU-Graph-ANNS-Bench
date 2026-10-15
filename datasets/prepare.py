"""Convert the downloaded raw files into one directory per dataset under $DATA_DIR.

Each dataset directory holds
  base.fbin, query.fbin    vectors (fbin)
  base.fvecs, query.fvecs  the same vectors for builders that only read fvecs
  gt.ibin                  ids of the 100 exact nearest neighbors of every query
  learn.fbin               MNIST8M only: IVF-PQ training vectors (elsewhere the first 10% of the base is used)
  README.txt               source and processing

Angular datasets (GloVe200, NYTimes) are L2-normalized and then searched with L2, which ranks neighbors as cosine
does. All-zero vectors have no cosine and are removed (NYTimes: 239 base vectors and 9 queries); the ground truth is
then recomputed. Datasets without a usable ground truth get one from groundtruth.py.

Usage: python datasets/prepare.py sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from binio import read_bin, write_bin, write_vecs

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent

# name -> (hdf5 file, angular)
ANN_BENCHMARKS = {
    'sift1M': ('sift-128-euclidean.hdf5', False),
    'gist': ('gist-960-euclidean.hdf5', False),
    'glove200': ('glove-200-angular.hdf5', True),
    'nytimes': ('nytimes-256-angular.hdf5', True),
}
DEEP1B_PREFIX = {'deep1M': 1_000_000, 'deep10M': 10_000_000, 'deep100M': 100_000_000}
# Variants used by some experiments, see datasets/README.md
DEEP_IMAGE = {'deep10M-annb': None, 'deep1M-annb': 1_000_000}
MNIST8M_TOTAL, MNIST8M_QUERY, MNIST8M_DIM = 8_100_000, 10_000, 784


def write_dataset(out: Path, base: np.ndarray, query: np.ndarray, gt, note: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_bin(out / 'base.fbin', base)
    write_bin(out / 'query.fbin', query)
    write_vecs(out / 'base.fvecs', base)
    write_vecs(out / 'query.fvecs', query)
    if gt is not None:
        write_bin(out / 'gt.ibin', gt, np.int32)
    else:
        (out / 'gt.ibin').unlink(missing_ok=True)
    (out / 'README.txt').write_text(f'{note}base {base.shape}, query {query.shape}\n'
                                    f"gt: {'from source ' + str(gt.shape) if gt is not None else 'groundtruth.py'}\n")
    logger.info('%s: base %s, query %s', out.name, base.shape, query.shape)


def normalize(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return (a / norms).astype(np.float32)


def ann_benchmarks(raw: Path, out: Path, name: str, keep_zero: bool = False) -> None:
    fname, angular = ANN_BENCHMARKS[name]
    with h5py.File(raw / fname, 'r') as f:
        base = f['train'][:].astype(np.float32)
        query = f['test'][:].astype(np.float32)
        gt = f['neighbors'][:].astype(np.int32)
    note = f'source: ann-benchmarks {fname}\n'
    if angular and not keep_zero:
        keep_b = np.where(np.linalg.norm(base, axis=1) > 0)[0]
        keep_q = np.where(np.linalg.norm(query, axis=1) > 0)[0]
        if keep_b.size < base.shape[0] or keep_q.size < query.shape[0]:
            note += (f'removed {base.shape[0] - keep_b.size} all-zero base vectors and '
                     f'{query.shape[0] - keep_q.size} all-zero queries\n')
            gt = None
        base, query = normalize(base[keep_b]), normalize(query[keep_q])
        note += 'L2-normalized; search with L2\n'
    elif angular:
        note += 'raw vectors including all-zero ones; search with cosine\n'
    write_dataset(out, base, query, gt, note)


def deep1b_prefix(raw: Path, out: Path, name: str) -> None:
    n, d = DEEP1B_PREFIX[name], 96
    base = np.fromfile(raw / f'deep1b_first{name[4:]}.fbin', dtype=np.float32, offset=8)
    if base.size != n * d:
        raise ValueError(f'{name}: got {base.size} floats, expected {n * d}')
    query = read_bin(raw / 'deep1b_query.public.10K.fbin')
    write_dataset(out, base.reshape(n, d), query, None,
                  f'source: first {n:,} vectors of Deep1B base.1B.fbin, official query.public.10K.fbin\n')


def deep_image(raw: Path, out: Path, name: str) -> None:
    with h5py.File(raw / 'deep-image-96-angular.hdf5', 'r') as f:
        n = DEEP_IMAGE[name] or f['train'].shape[0]
        base = f['train'][:n].astype(np.float32)
        query = f['test'][:].astype(np.float32)
    write_dataset(out, base, query, None,
                  f'source: first {n:,} vectors of ann-benchmarks deep-image-96-angular.hdf5 (unit vectors), its queries\n')


def mnist8m(raw: Path, out: Path) -> None:
    """MNIST8M: 784-dim pixel values / 255. The last 10,000 vectors are the queries, the rest the base."""
    with h5py.File(raw / 'MNIST8M_data.h5', 'r') as f:
        key = 'MNIST8M' if 'MNIST8M' in f else list(f.keys())[0]
        ds = f[key]
        if ds.shape != (MNIST8M_DIM, MNIST8M_TOTAL):
            raise ValueError(f'unexpected shape {ds.shape}')
        n_base = MNIST8M_TOTAL - MNIST8M_QUERY
        base = np.empty((n_base, MNIST8M_DIM), dtype=np.float32)
        for s in range(0, n_base, 100_000):
            e = min(s + 100_000, n_base)
            base[s:e] = np.asarray(ds[:, s:e], dtype=np.float32).T * np.float32(1.0 / 255.0)
        query = np.asarray(ds[:, n_base:], dtype=np.float32).T * np.float32(1.0 / 255.0)
    write_dataset(out, base, query, None,
                  'source: MNIST8M_data.h5 (LIBSVM mnist8m as uint8), values / 255; queries are the last 10,000\n'
                  'learn.fbin: the first 100,000 base vectors (IVF-PQ training set)\n')
    write_bin(out / 'learn.fbin', base[:100_000])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='+')
    ap.add_argument('--data-dir', default=os.environ.get('DATA_DIR', 'data'))
    ap.add_argument('--gpu', action='store_true', help='compute missing ground truth on the GPU')
    args = ap.parse_args()
    root = Path(args.data_dir)
    raw = root / 'raw'
    for name in args.names:
        out = root / name
        if name in ANN_BENCHMARKS:
            ann_benchmarks(raw, out, name)
        elif name == 'nytimes-full':
            ann_benchmarks(raw, out, 'nytimes', keep_zero=True)
        elif name in DEEP1B_PREFIX:
            deep1b_prefix(raw, out, name)
        elif name in DEEP_IMAGE:
            deep_image(raw, out, name)
        elif name == 'mnist8M':
            mnist8m(raw, out)
        else:
            sys.exit(f'unknown dataset {name}')
        if not (out / 'gt.ibin').exists():
            cmd = [sys.executable, str(HERE / 'groundtruth.py'), str(out)] + (['--gpu'] if args.gpu else [])
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
