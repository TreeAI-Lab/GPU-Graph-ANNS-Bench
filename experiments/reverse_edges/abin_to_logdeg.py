"""Convert graph.abin into the input format of make_graph_to_cagra.

graph.abin (written by cagra_build_csr): int32 n, int32 degree, then n x degree uint32 neighbor ids.
Output: uint32 n, uint32 log2(degree), then the same neighbor ids.
"""
import argparse
import math

import numpy as np


def convert(src: str, dst: str) -> None:
    """Rewrite the header of a fixed-degree graph; the degree must be a power of two."""
    with open(src, 'rb') as f:
        n, deg = np.fromfile(f, dtype=np.int32, count=2)
        edges = np.fromfile(f, dtype=np.uint32)
    assert edges.size == int(n) * int(deg), f'{src}: {edges.size} ids for n={n} degree={deg}'
    logd = int(math.log2(int(deg)))
    assert 1 << logd == int(deg), f'{src}: degree {deg} is not a power of two'
    with open(dst, 'wb') as f:
        f.write(np.array([n, logd], dtype=np.uint32).tobytes())
        f.write(edges.astype(np.uint32).tobytes())
    print('wrote', dst, 'n', n, 'deg', deg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('abin', help='graph.abin from cagra_build_csr')
    ap.add_argument('out', help='output file for make_graph_to_cagra')
    args = ap.parse_args()
    convert(args.abin, args.out)


if __name__ == '__main__':
    main()
