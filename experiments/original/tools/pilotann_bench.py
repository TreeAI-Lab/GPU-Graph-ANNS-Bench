"""Search a PilotANN index with the profiler of PilotANN (script/common.py, ANNProfiler).

The profiler searches n_queries random queries; here they are drawn once per process, so that every search
searches the same queries. For every ef_search the profiler runs WARMUP_RUNS untimed searches and TIMED_RUNS
timed searches and prints the mean recall and the mean time of one timed search; this script then prints the
number of timed searches and the standard deviation of their times. The dataset directory holds base.fvecs,
query.fvecs and groundtruth.ivecs (the layout read by the patched PilotANN loader).

Runs in the pilot environment. Usage:
  python3 pilotann_bench.py --dataset_dir DIR --checkpoint INDEX.ckpt --algorithm nsg --top_k 10
      --n_queries 1024 --ef_search "[30, 32]" [--pilotann_src $W/src/pilotann]
"""
import argparse
import ast
import os
import statistics
import sys
import time
from typing import Callable, List

WARMUP_RUNS = 1
TIMED_RUNS = 3


def record_durations(search: Callable, durations: List[float]) -> Callable:
    """Wraps a timed search method of the profiler (it returns the output and the time in s) to keep the times."""
    def wrapped(*args, **kwargs):
        output, duration = search(*args, **kwargs)
        durations.append(1000.0 * duration)
        return output, duration
    return wrapped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='directory with base/query fvecs and gt')
    parser.add_argument('--checkpoint', type=str, required=True, help='index file (.ckpt or .faiss)')
    parser.add_argument('--pilotann_src', type=str,
                        default=os.path.join(os.environ.get('W', ''), 'src', 'pilotann'),
                        help='PilotANN source tree (for script/common.py)')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--ef_search', type=str, default='[16, 32, 48, 64, 96, 128, 192, 256]')
    parser.add_argument('--algorithm', type=str, default='hnsw', choices=['hnsw', 'nsg'])
    parser.add_argument('--n_queries', type=int, default=1024)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.pilotann_src, 'script'))
    from common import ANNProfiler

    ef_searchs = ast.literal_eval(args.ef_search)
    method = os.path.basename(args.checkpoint).split('.')[-1]
    profiler = ANNProfiler(
        dataset=args.dataset_dir,
        algorithm=args.algorithm,
        n_queries=args.n_queries,
        entry_method=None,
        sample_ratio=None,
        d_principle=None
    )
    profiler.N_WARMUP = WARMUP_RUNS
    sample = profiler.loader.load_query(n_queries=args.n_queries, k=args.top_k)
    profiler.loader.load_query = lambda n_queries, k: sample
    durations: List[float] = []
    for name in ('search_faiss', 'search_pilot', 'search_hnswlib'):
        if hasattr(profiler, name):
            setattr(profiler, name, record_durations(getattr(profiler, name), durations))

    records = []
    profiler.load(args.checkpoint)
    for ef_search in ef_searchs:
        time.sleep(2.0)
        print(args.top_k, ef_search)
        durations.clear()
        recall, timing = profiler.run(k=args.top_k, ef_search=ef_search, n_repeats=TIMED_RUNS)
        print('{}: recall={:.3f}, duration={:.3f}ms'.format(method, recall, timing))
        timed = durations[WARMUP_RUNS:]
        std = statistics.stdev(timed) if len(timed) > 1 else 0.0
        print('n_timed={}, duration_std={:.3f}ms'.format(len(timed), std))
        records.append({'recall': recall, 'timing': timing, 'ef_search': ef_search})
    print(method, records)


if __name__ == '__main__':
    main()
