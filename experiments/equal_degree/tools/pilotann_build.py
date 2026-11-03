"""Build a PilotANN index with the profiler of PilotANN (script/common.py, ANNProfiler) and save it.

Prints the build time (Tables 5 and 6). method search-pilot builds the staged index (IndexStaged) on the
graph <algorithm><n_neighbors>; method faiss builds a Faiss HNSW (M = n_neighbors / 2) or NSG index.
The dataset directory holds base.fvecs, query.fvecs and groundtruth.ivecs.

Runs in the pilot environment. Usage:
  python3 pilotann_build.py --dataset_dir DIR --output INDEX.ckpt --method search-pilot --algorithm nsg
      --n_neighbors 32 [--d_principle 96 --sample_ratio 0.25 --router router32x32]
"""
import argparse
import os
import sys
from time import perf_counter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='directory with base/query fvecs and gt')
    parser.add_argument('--output', type=str, required=True, help='index file to write')
    parser.add_argument('--pilotann_src', type=str,
                        default=os.path.join(os.environ.get('W', ''), 'src', 'pilotann'),
                        help='PilotANN source tree (for script/common.py)')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--method', type=str, default='faiss')
    parser.add_argument('--d_principle', type=int, default=96)
    parser.add_argument('--sample_ratio', type=float, default=0.25)
    parser.add_argument('--n_neighbors', type=int, default=32)
    parser.add_argument('--n_queries', type=int, default=1024)
    parser.add_argument('--router', type=str, default='router32x32')
    parser.add_argument('--algorithm', type=str, default='hnsw', choices=['hnsw', 'nsg'])
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.pilotann_src, 'script'))
    from common import ANNProfiler

    profiler = ANNProfiler(
        dataset=args.dataset_dir,
        algorithm=args.algorithm,
        entry_method=args.router,
        sample_ratio=args.sample_ratio,
        d_principle=args.d_principle,
        n_queries=args.n_queries
    )

    t0 = perf_counter()
    profiler.build(method=args.method, n_neighbors=args.n_neighbors)
    t1 = perf_counter()
    print(f'Build completed in {t1 - t0:.3f} seconds')

    profiler.dump(checkpoint=args.output)
    print('done')


if __name__ == '__main__':
    main()
