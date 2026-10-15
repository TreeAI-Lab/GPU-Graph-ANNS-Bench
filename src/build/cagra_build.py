"""CAGRA build driver (cuVS 25.02.01).

NN-descent is chosen explicitly: the Python interface defaults to IVF-PQ for the initial kNN graph (cagra.pyx:178),
while the C++ interface chooses NN-descent when GPU memory suffices (cagra_build.cuh:593-603).
CAGRA_BUILD_SECONDS starts with copying the base vectors to GPU memory, as the timed windows of GGNN and GANNS do
(both upload the data inside them); CUDA initialization is outside the window, as for those two. Copying the graph
back to the host is not inside the window: the Python Index of 25.02 does not expose the graph, which can only be
read from the file written by cagra.save, and writing files is not timed for any builder.
Usage: python cagra_build.py <base.fbin> <out.idx> <graph_degree> <intermediate_graph_degree> <nn_descent_niter>
"""
import argparse
import time

import numpy as np
from cuvs.neighbors import cagra
from pylibraft.common import device_ndarray


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("graph_degree", type=int)
    ap.add_argument("intermediate_graph_degree", type=int)
    ap.add_argument("nn_descent_niter", type=int)
    a = ap.parse_args()
    gd, igd, niter = a.graph_degree, a.intermediate_graph_degree, a.nn_descent_niter
    n, d = np.fromfile(a.base, dtype=np.int32, count=2)
    x = np.fromfile(a.base, dtype=np.float32, offset=8).reshape(int(n), int(d))
    device_ndarray(np.zeros(1, dtype=np.float32))  # CUDA initialization outside the timed window
    t0 = time.time()
    xd = device_ndarray(x)
    idx = cagra.build(cagra.IndexParams(metric="sqeuclidean", graph_degree=gd, intermediate_graph_degree=igd,
                                        build_algo="nn_descent", nn_descent_niter=niter), xd)
    print(f"CAGRA_BUILD_SECONDS {time.time() - t0:.3f}")
    cagra.save(a.out, idx, include_dataset=False)


if __name__ == "__main__":
    main()
