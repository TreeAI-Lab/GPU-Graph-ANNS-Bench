"""Build a large query file for the serving experiment on DEEP100M: the 10,000 official queries followed by
vectors drawn from Deep1B beyond the first 100M, so none of them is in the DEEP100M base.

Vectors are fetched by HTTP range requests in aligned blocks of BLOCK consecutive vectors; blocks are drawn at random
without replacement and all drawn rows are shuffled with the same seed.

Output (fbin):
  <out>              the official queries first, then the drawn vectors
  <out>.src_ids.ibin the id of every row in base.1B.fbin, -1 for the official queries, header (n, 1)

Usage:
  python datasets/sample_deep1b_queries.py --official query.public.10K.fbin --out query.5M.fbin --total 5000000
  python datasets/sample_deep1b_queries.py --validate base.100M.fbin   # compare 20 blocks with a local copy
"""
import argparse
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DIM = 96
VEC_BYTES = DIM * 4
HEADER = 8
URL = os.environ.get("DEEP1B_URL", "https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP") + "/base.1B.fbin"

logger = logging.getLogger(__name__)


def fetch_block(start: int, n: int, url: str, retries: int = 6) -> np.ndarray:
    """Vectors with ids [start, start + n) as an (n, DIM) float32 array."""
    off = HEADER + start * VEC_BYTES
    req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + n * VEC_BYTES - 1}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                crange = r.headers.get("Content-Range", "")
                if r.status != 206 or not crange.startswith(f"bytes {off}-"):
                    raise IOError(f"HTTP {r.status}, Content-Range {crange!r}")
                data = r.read()
            if len(data) != n * VEC_BYTES:
                raise IOError(f"got {len(data)} bytes, expected {n * VEC_BYTES}")
            return np.frombuffer(data, dtype=np.float32).reshape(n, DIM)
        except (urllib.error.URLError, IOError, TimeoutError) as e:
            wait = 2 ** attempt
            logger.warning("block at id %d failed (%s), retry in %ds", start, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"block at id {start} failed after {retries} attempts")


def fetch_blocks(starts: np.ndarray, block: int, url: str, workers: int) -> np.ndarray:
    """Fetch the blocks in parallel and concatenate them in the order of starts."""
    out = np.empty((len(starts) * block, DIM), dtype=np.float32)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for i, vecs in enumerate(ex.map(lambda s: fetch_block(int(s), block, url), starts)):
            out[i * block:(i + 1) * block] = vecs
            if i % 250 == 0 or i == len(starts) - 1:
                el = time.time() - t0
                logger.info("%d / %d blocks, %.0f s, ETA %.0f s", i + 1, len(starts), el, el / (i + 1) * (len(starts) - i - 1))
    return out


def read_fbin(path: Path) -> np.ndarray:
    n, d = np.fromfile(path, dtype=np.int32, count=2)
    return np.memmap(path, dtype=np.float32, mode="r", offset=HEADER, shape=(int(n), int(d)))


def write_bin(path: Path, arr: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.array(arr.shape, dtype=np.int32).tofile(f)
        arr.tofile(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=URL)
    ap.add_argument("--official", type=Path, help="official query file, written first")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--total", type=int, default=5_000_000, help="total rows including the official queries")
    ap.add_argument("--lo", type=int, default=100_000_000, help="lowest id to draw from, a multiple of --block")
    ap.add_argument("--hi", type=int, default=1_000_000_000, help="id bound (exclusive), a multiple of --block")
    ap.add_argument("--block", type=int, default=1000, help="consecutive vectors per block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--validate", type=Path, help="local base.100M.fbin: compare 20 blocks with it and exit")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    if a.validate:
        base = read_fbin(a.validate)
        starts = np.random.default_rng(12345).choice(base.shape[0] // a.block, size=20, replace=False) * a.block
        got = fetch_blocks(starts, a.block, a.url, a.workers)
        want = np.concatenate([base[s:s + a.block] for s in starts])
        bad = int((got != want).any(axis=1).sum())
        logger.info("validate: %d of %d vectors differ from %s", bad, len(got), a.validate)
        raise SystemExit(1 if bad else 0)

    if a.lo % a.block or a.hi % a.block:
        raise ValueError("--lo and --hi must be multiples of --block")
    official = np.asarray(read_fbin(a.official))
    n_new = a.total - official.shape[0]
    if n_new % a.block:
        raise ValueError(f"total - n_official = {n_new} is not a multiple of block {a.block}")
    rng = np.random.default_rng(a.seed)
    starts = np.sort(rng.choice((a.hi - a.lo) // a.block, size=n_new // a.block, replace=False)) * a.block + a.lo
    logger.info("%d blocks of %d from [%d, %d), seed %d; %d official queries first",
                len(starts), a.block, a.lo, a.hi, a.seed, official.shape[0])
    vecs = fetch_blocks(starts, a.block, a.url, a.workers)
    ids = (starts[:, None] + np.arange(a.block)[None, :]).reshape(-1)
    perm = rng.permutation(n_new)
    write_bin(a.out, np.concatenate([official, vecs[perm]]))
    src = np.concatenate([np.full(official.shape[0], -1, dtype=np.int64), ids[perm]]).astype(np.int32).reshape(-1, 1)
    write_bin(Path(f"{a.out}.src_ids.ibin"), src)
    logger.info("wrote %s (%d x %d) and %s.src_ids.ibin", a.out, a.total, DIM, a.out)


if __name__ == "__main__":
    main()
