"""Build one PilotANN index (IndexStaged: SVD, full graph, sampled subgraph) and save its checkpoint.

ef_construct is not exposed by IndexStaged, so the default argument of graph_init is replaced by
--efc for both calls (full graph and sampled subgraph); the PilotANN sources are not changed.

Used for the PilotANN indexes of Figures 7, 8 and 12.
Usage (environment $ENV_PILOT):
  python src/search/pilotann/pilotann_build.py --base base.fbin --ckpt out.pt --nn 64 [--efc 80] [--dp 192]
      [--ratio 0.25] [--gtype nsg]
"""
import argparse
import functools
import time
from pathlib import Path

import numpy as np
import torch
from pilot_ann import layers, utils
from pilot_ann.utils import graphs as pilot_graphs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base vectors, fbin")
    ap.add_argument("--ckpt", type=Path, required=True, help="output checkpoint (.pt)")
    ap.add_argument("--nn", type=int, required=True, help="graph degree (n_neighbors)")
    ap.add_argument("--efc", type=int, default=80, help="ef_construct of the graph builder")
    ap.add_argument("--dp", type=int, default=192, help="d_principle, capped at the dimension")
    ap.add_argument("--ratio", type=float, default=0.25, help="sample_ratio of the GPU subgraph")
    ap.add_argument("--gtype", default="nsg", help="graph method: nsg or nsw")
    args = ap.parse_args()

    n, d = np.fromfile(args.base, dtype=np.int32, count=2)
    x = torch.from_numpy(np.fromfile(args.base, dtype=np.float32, offset=8).reshape(int(n), int(d)).copy())
    dp = min(args.dp, int(d))
    print(f"train n={int(n)} d={int(d)} dp={dp} {args.gtype}{args.nn} efc={args.efc} r={args.ratio}", flush=True)
    orig = pilot_graphs.graph_init
    utils.graph_init = pilot_graphs.graph_init = functools.partial(orig, ef_construct=args.efc)
    index = layers.IndexStaged(
        d_model=int(d),
        d_principle=dp,
        sample_ratio=args.ratio,
        graph_method=f"{args.gtype}{args.nn}",
    )
    t0 = time.time()
    index.train(x)
    print(f"PILOT_BUILD_SECONDS {time.time() - t0:.3f}", flush=True)
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index.dump(), args.ckpt)
    print("saved", args.ckpt, flush=True)


if __name__ == "__main__":
    main()
