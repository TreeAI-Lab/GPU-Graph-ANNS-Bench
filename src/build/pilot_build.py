"""PilotANN build driver: runs the official IndexStaged.train (SVD, full graph, sampled subgraph) and stores the
full graph as npz. PILOT_BUILD_SECONDS is the time of the whole train call.

IndexStaged does not expose ef_construct, so the default of graph_init is replaced by the given value (for both
calls, full graph and subgraph); the PilotANN sources are not changed.
Usage: python pilot_build.py <base.fbin> <out.npz> <graph_type> <n_neighbors> <ef_construct> <d_principle> <sample_ratio>
"""
import argparse
import functools
import time

import numpy as np
import torch
from pilot_ann import layers, utils
from pilot_ann.utils import graphs as pilot_graphs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("graph_type", help="graph method prefix of PilotANN, e.g. nsg")
    ap.add_argument("n_neighbors", type=int)
    ap.add_argument("ef_construct", type=int)
    ap.add_argument("d_principle", type=int)
    ap.add_argument("sample_ratio", type=float)
    a = ap.parse_args()
    gtype, nn_, efc, dp, ratio = a.graph_type, a.n_neighbors, a.ef_construct, a.d_principle, a.sample_ratio
    n, d = np.fromfile(a.base, dtype=np.int32, count=2)
    x = torch.from_numpy(np.fromfile(a.base, dtype=np.float32, offset=8).reshape(int(n), int(d)).copy())
    # The full graph is built through utils.graph_init, the subgraph through graph_init inside the graphs module;
    # both get the same ef_construct.
    _orig = pilot_graphs.graph_init
    utils.graph_init = pilot_graphs.graph_init = functools.partial(_orig, ef_construct=efc)
    index = layers.IndexStaged(d_model=int(d), d_principle=min(dp, int(d)), sample_ratio=ratio, graph_method=f"{gtype}{nn_}")
    t0 = time.time()
    index.train(x)
    print(f"PILOT_BUILD_SECONDS {time.time() - t0:.3f}")
    fg = index.fullgraph
    np.savez(a.out, indptr=np.asarray(fg.indptr.cpu()), indices=np.asarray(fg.indices.cpu()))


if __name__ == "__main__":
    main()
