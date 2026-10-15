"""Exact top-k ground truth of a large query file (the 5,000,000 queries of Figure 6), with cuVS brute force.

The base set is kept on the GPU as one brute-force index; the queries are searched in batches.
Output: ibin, int32 n, int32 k, then n x k int32 neighbor ids (squared L2). A log is written next to it
(<out> with suffix .log); its last line starts with "DONE wrote" when the file is complete.

Usage (environment $ENV_CAGRA):
  python gt_5m.py --base base.fbin --query query.5M.fbin --out gt.5M.k10.ibin --index bf.cuvsindex
"""
import argparse
import struct
import time
from pathlib import Path

import numpy as np
from cuvs.neighbors import brute_force
from pylibraft.common import device_ndarray


def load_base(path: str) -> np.ndarray:
    """fbin or fvecs as an (n, dim) float32 array."""
    if path.endswith(".fbin"):
        with open(path, "rb") as f:
            nb, dim = struct.unpack("<ii", f.read(8))
            return np.fromfile(f, dtype=np.float32).reshape(nb, dim)
    with open(path, "rb") as f:
        dim = struct.unpack("<i", f.read(4))[0]
    raw = np.fromfile(path, dtype=np.float32)
    assert raw.size % (dim + 1) == 0
    nb = raw.size // (dim + 1)
    return np.ascontiguousarray(raw.reshape(nb, dim + 1)[:, 1:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base vectors, fbin or fvecs")
    ap.add_argument("--query", required=True, help="query fbin")
    ap.add_argument("--out", required=True, help="output ibin")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--index", required=True, help="brute-force index file, built and saved if missing")
    ap.add_argument("--rebuild-index", action="store_true", help="ignore an existing index and rebuild")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")

    def logf(msg: str) -> None:
        line = f"{time.strftime('%F %T')} {msg}"
        print(line, flush=True)
        with log.open("a") as f:
            f.write(line + "\n")

    logf(f"loading base {args.base}")
    base = load_base(args.base)
    nb, dim = base.shape
    logf(f"base n={nb} dim={dim}")

    with open(args.query, "rb") as f:
        nq, qd = struct.unpack("<ii", f.read(8))
    assert qd == dim
    logf(f"query n={nq} dim={qd} batch={args.batch} k={args.k}")

    # CUDA initialization, then build or load the brute-force index (resident on the GPU)
    device_ndarray(np.zeros(1, dtype=np.float32))
    t0 = time.time()
    idx_path = Path(args.index)
    if idx_path.is_file() and not args.rebuild_index:
        logf(f"loading brute-force index {args.index}")
        index = brute_force.load(str(idx_path))
    else:
        logf("building brute-force index on the device from this base")
        xd = device_ndarray(base)
        index = brute_force.build(xd, metric="sqeuclidean")
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        brute_force.save(str(idx_path), index)
        logf(f"saved brute-force index {args.index}")
        # Reload so that the index owns its serialized dataset
        index = brute_force.load(str(idx_path))
        del xd
    logf(f"index ready in {time.time() - t0:.2f}s")
    del base

    nbytes = 8 + nq * args.k * 4
    logf(f"allocating output {out} ({nbytes / 1e9:.2f} GB)")
    with open(out, "wb") as f:
        f.write(struct.pack("<ii", nq, args.k))
        f.seek(nbytes - 1)
        f.write(b"\0")

    mm = np.memmap(out, dtype=np.int32, mode="r+", offset=8, shape=(nq, args.k))

    nbatch = (nq + args.batch - 1) // args.batch
    t_all = time.time()
    with open(args.query, "rb") as qf:
        qf.seek(8)
        for bi in range(nbatch):
            s = bi * args.batch
            e = min(nq, s + args.batch)
            bsz = e - s
            q = np.fromfile(qf, dtype=np.float32, count=bsz * dim).reshape(bsz, dim)
            t1 = time.time()
            qd_dev = device_ndarray(np.ascontiguousarray(q))
            dist, neigh = brute_force.search(index, qd_dev, args.k)
            ids = neigh.copy_to_host().astype(np.int32).reshape(bsz, args.k)
            mm[s:e] = ids
            mm.flush()
            del qd_dev, dist, neigh
            dt = time.time() - t1
            if bi % 10 == 0 or bi + 1 == nbatch:
                logf(f"batch {bi + 1}/{nbatch} q[{s}:{e}] {dt:.3f}s  qps={bsz / max(dt, 1e-9):.0f}  "
                     f"elapsed={time.time() - t_all:.1f}s")

    del mm
    logf(f"DONE wrote {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
