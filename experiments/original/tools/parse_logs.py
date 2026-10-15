"""Turn the logs of the run and build scripts of this folder into the tables used by the figures.

Every run directory ($OUT_DIR/original/<system>/<run>/k<K>/) holds meta.tsv (system, dataset, k, nq) and one
log per search setting. Subcommands:

  curves      recall and QPS of every log (Figures 7, 11, 12, 13)
  stages      share of search time per stage at one operating point per run (Figure 9)
  iterations  visited nodes per query at the same operating point (Figure 5(a)); the path counter of each
              program, except CAGRA (max_iterations per query) and SONG (its counter starts at 100)
  build       construction time, peak host memory, average degree and components (Tables 5 and 6)

The operating point of a run is the log whose recall is closest to --target (default 0.9). stages and
iterations use the runs at the default degree (run name = dataset) and K = --k (default 10) unless --all-runs.

Warm-up and timed runs. Each setting is searched once as an untimed warm-up and then three times; a record is
the mean of the timed runs: times and counters are means, QPS is nq / mean time, recall is the mean recall of the
timed runs, total_time_us_std is the sample standard deviation of the time of the timed runs and n_timed their
number.
  SONG, GGNN, GANNS  one process per run; the log of a setting holds all runs, each after a marker line
                     "### run warmup <i>/<n>" or "### run timed <i>/<n>" (repeat_runs in common.sh), and the
                     .recall file one line per timed run
  BANG               bang_search repeats every L itself: tagged lines "[warmup i/n]" and "[timed i/n]" after
                     every search, then the summary line (mean of the timed searches)
  PilotANN           pilotann_bench.py prints the mean of the timed searches, n_timed and duration_std
  Vamana             one search_disk_index process per L with L repeated; the first line of the log gives the
                     number of warm-up rows, which are dropped
  CAGRA              my_cagra prints no time; one run per setting
A log without these markers is read as one timed run.

Stage groups (CL: candidate locating, DC: distance calculation, DM: data maintenance):
  SONG      the three printed stage percentages, in this order
  GGNN      fetch and filter, calc, update (printed as ratios)
  GANNS     six printed stage shares: CL = 1 + 2 (pick the next candidate, read its neighbor list),
            DC = 3, DM = 4 + 5 + 6 (duplicate check, sort, merge)
  CAGRA     clocks of a cuVS build with _CLK_BREAKDOWN: CL = pickup_parents, DC = 1st_distance + distance,
            DM = topk + reset_hash + restore_hash (summed over the printed queries)
  BANG      STATS of the patched bang_search in ms: GPU CL = (4) neighbor filtering + (5) prefetch,
            GPU DC = (1) + (2) + (6), GPU DM = (3) sort and merge, CPU CL = (8) neighbor seek
The GANNS, CAGRA and BANG groupings are read from the kernels; no file records the grouping of the
original table.

Usage: python3 parse_logs.py curves|stages|iterations|build [--root DIR] [--out FILE.csv] [--target 0.9]
"""
import argparse
import csv
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

STAGES = ['gpu_cl', 'cpu_cl', 'gpu_dc', 'cpu_dc', 'gpu_dm', 'cpu_dm']
NAME_FIG05 = {'sift1M': 'SIFT1M', 'gist': 'GIST', 'deep1M': 'DEEP1M', 'deep10M': 'DEEP10M'}
SYSTEM = {'song': 'SONG', 'ggnn': 'GGNN', 'ganns': 'GANNS', 'cagra': 'CAGRA', 'bang': 'BANG',
          'pilotann': 'PilotANN', 'vamana': 'Vamana'}
NUM = r'([-+0-9.eE]+|nan|inf)'
RUN_MARK = re.compile(r'^### run (warmup|timed) (\d+)/(\d+)\s*$', flags=re.M)


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def stdev(xs: Sequence[float]) -> float:
    """Sample standard deviation (n - 1 denominator), 0 for one value."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def first(pattern: str, text: str, cast=float, last: bool = False) -> Optional[float]:
    found = re.findall(pattern, text, flags=re.M)
    if not found:
        return None
    return cast(found[-1] if last else found[0])


def recall_file(log: str) -> Optional[float]:
    """Mean of the recall lines of <log>.recall (one line per timed run)."""
    path = os.path.splitext(log)[0] + '.recall'
    if not os.path.exists(path):
        return None
    found = [float(x) for x in re.findall(r'Recall@\d+ = ' + NUM, open(path).read())]
    return mean(found) if found else None


def timed_sections(text: str) -> Optional[List[str]]:
    """Output of the timed runs of a log written by repeat_runs (common.sh); None for a log without run markers."""
    marks = list(RUN_MARK.finditer(text))
    if not marks:
        return None
    out = []
    for i, m in enumerate(marks):
        if m.group(1) == 'timed':
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            out.append(text[m.end():end])
    return out


def merge_runs(runs: List[List[dict]]) -> List[dict]:
    """One record per setting from the records of the timed runs (records in the same order in every run).

    Numbers are averaged, stage shares stage by stage; QPS is nq / mean time, that is the harmonic mean of the
    QPS of the runs."""
    merged = []
    for recs in zip(*runs):
        rec = dict(recs[0])
        for key in recs[0]:
            vals = [r.get(key) for r in recs]
            if key == 'stages':
                st = [v for v in vals if v]
                rec[key] = {s: mean([v[s] for v in st]) for s in STAGES}
            elif key == 'qps':
                if all(isinstance(v, (int, float)) and v > 0 for v in vals):
                    rec[key] = len(vals) / sum(1.0 / v for v in vals)
            elif all(isinstance(v, (int, float)) for v in vals):
                rec[key] = mean(vals)
        times = [r.get('total_time_us') for r in recs]
        if all(isinstance(t, (int, float)) for t in times):
            rec['total_time_us_std'] = stdev(times)
        rec['n_timed'] = len(recs)
        merged.append(rec)
    return merged


def shares(raw: Dict[str, float]) -> Dict[str, float]:
    total = sum(raw.get(s, 0.0) for s in STAGES)
    return {s: (raw.get(s, 0.0) / total if total else float('nan')) for s in STAGES}


def parse_song(log: str, text: str, nq: int) -> List[dict]:
    kernel_us = first(r'kernel takes (\d+) microseconds', text)
    st = re.search(r'stages percentage ' + NUM + ' ' + NUM + ' ' + NUM, text)
    rec = {'recall': recall_file(log), 'path_total': first(r'^path_length: (\d+)', text, int),
           'calc_total': first(r'^calculate_count: (\d+)', text, int), 'total_time_us': kernel_us,
           'qps': nq / kernel_us * 1e6 if kernel_us else None}
    if st:
        a, b, c = (float(x) for x in st.groups())
        rec['stages'] = shares({'gpu_cl': a, 'gpu_dc': b, 'gpu_dm': c})
    return [rec]


def parse_ggnn(log: str, text: str, nq: int) -> List[dict]:
    kernel_ms = [float(x) for x in re.findall(r'GPU kernel time: ' + NUM + ' ms', text)]
    total_us = sum(kernel_ms) * 1000 if kernel_ms else first(r'Query completed in ' + NUM + ' us', text)
    rec = {'recall': recall_file(log), 'path_total': first(r'^path count: (\d+)', text, int),
           'calc_total': first(r'^calc count: (\d+)', text, int), 'total_time_us': total_us,
           'qps': nq / total_us * 1e6 if total_us else None}
    fetch = first(r'fetch time ratio: ' + NUM, text)
    if fetch is not None:
        rec['stages'] = shares({'gpu_cl': fetch, 'gpu_dc': first(r'calc time ratio: ' + NUM, text),
                                'gpu_dm': first(r'update time ratio: ' + NUM, text)})
    return [rec]


def parse_ganns(log: str, text: str, nq: int) -> List[dict]:
    rec = {'recall': first(r'^Recall: ' + NUM, text), 'qps': first(r'Query speed: ' + NUM, text),
           'total_time_us': first(r'^query time: (\d+) us', text),
           'path_total': first(r'^path_length: (\d+)', text, int),
           'calc_total': first(r'^cal_count: (\d+)', text, int), 'avg_degree': first(r'^Average degree: ' + NUM, text),
           'components': first(r'^Number of components: (\d+)', text, int)}
    st = re.search(r'stages percentage: (.+)', text)
    if st:
        s = [float(x) for x in st.group(1).split()]
        rec['stages'] = shares({'gpu_cl': s[0] + s[1], 'gpu_dc': s[2], 'gpu_dm': s[3] + s[4] + s[5]})
    return [rec]


CLK = re.compile(r'query, (\d+), thread, (\d+), init, (\d+), 1st_distance, (\d+), topk, (\d+), reset_hash, (\d+), '
                 r'pickup_parents, (\d+), restore_hash, (\d+), distance, (\d+)')


def parse_cagra(log: str, text: str, nq: int) -> List[dict]:
    """my_cagra prints no time and no counter; visited nodes are taken as max_iterations per query."""
    it = re.search(r'Using search iterations: (\d+)', text)
    rec = {'recall': recall_file(log), 'qps': None, 'total_time_us': None,
           'path_total': int(it.group(1)) * nq if it else None, 'path_from': 'max_iterations'}
    clk = [tuple(int(v) for v in m.groups()) for m in CLK.finditer(text)]
    clk = [c for c in clk if c[1] == 0]
    if clk:
        tot = [sum(c[i] for c in clk) for i in range(2, 9)]
        init, first_d, topk, reset, pickup, restore, dist = tot
        rec['stages'] = shares({'gpu_cl': pickup, 'gpu_dc': first_d + dist, 'gpu_dm': topk + reset + restore})
    return [rec]


BANG_RESULT = re.compile(r'^(\d+)\t' + NUM + r'\t' + NUM + r'\t' + NUM + r'((?:\t' + NUM + r')*)\s*$')
BANG_STAT = re.compile(r'^\((\d)\) [^=]*= ?' + NUM)
BANG_RUN = re.compile(r'^\[(warmup|timed) (\d+)/(\d+)\] L=(\d+) wall_ms=' + NUM)


def parse_bang(log: str, text: str, nq: int) -> List[dict]:
    """One record per summary line (L, mean wall ms, QPS, recall %, then optionally the standard deviation of the
    wall time and the number of timed searches). STATS and counters are the means over the searches tagged
    "[timed ...]" since the previous summary line; searches tagged "[warmup ...]" are dropped. Without tagged
    timed searches, the STATS of the search just before the summary line are used."""
    recs, stats, counts, timed = [], {}, {}, []
    for line in text.splitlines():
        m = BANG_STAT.match(line)
        if m:
            stats[int(m.group(1))] = float(m.group(2))
            continue
        m = re.match(r'calc_count=(\d+), path_length=(\d+)', line)
        if m:
            counts = {'calc_total': int(m.group(1)), 'path_total': int(m.group(2))}
            continue
        m = BANG_RUN.match(line)
        if m:
            if m.group(1) == 'timed':
                timed.append((stats, counts, float(m.group(5))))
            stats, counts = {}, {}
            continue
        m = BANG_RESULT.match(line)
        if m:
            runs = timed or [(stats, counts, None)]
            rec = {'param': 'L=' + m.group(1), 'total_time_us': float(m.group(2)) * 1000, 'qps': float(m.group(3)),
                   'recall': float(m.group(4)) / 100.0}
            for key in ('calc_total', 'path_total'):
                vals = [c[key] for _, c, _ in runs if key in c]
                if vals:
                    rec[key] = mean(vals)
            extra = [float(x) for x in m.group(5).split('\t')[1:]]
            walls = [w for _, _, w in timed]
            if extra:
                rec['total_time_us_std'] = extra[0] * 1000
            elif len(walls) > 1:
                rec['total_time_us_std'] = stdev(walls) * 1000
            if len(extra) > 1:
                rec['n_timed'] = int(extra[1])
            elif timed:
                rec['n_timed'] = len(timed)
            run_stats = [st for st, _, _ in runs if st]
            if run_stats:
                s = {i: mean([st.get(i, 0.0) for st in run_stats]) for i in range(1, 9)}
                rec['stages'] = shares({'gpu_cl': s[4] + s[5], 'gpu_dc': s[1] + s[2] + s[6],
                                        'gpu_dm': s[3], 'cpu_cl': s[8]})
            recs.append(rec)
            stats, counts, timed = {}, {}, []
    return recs


def parse_pilotann(log: str, text: str, nq: int) -> List[dict]:
    """Mean recall and mean time of the timed searches per ef, as printed by pilotann_bench.py."""
    recs = []
    pattern = (r'^\d+ (\d+)\n(?:.*\n)*?\w+: recall=' + NUM + ', duration=' + NUM + 'ms'
               r'(?:\nn_timed=(\d+), duration_std=' + NUM + 'ms)?')
    for m in re.finditer(pattern, text, flags=re.M):
        dur_ms = float(m.group(3))
        rec = {'param': 'ef=' + m.group(1), 'recall': float(m.group(2)), 'qps': nq / dur_ms * 1000,
               'total_time_us': dur_ms * 1000}
        if m.group(4):
            rec['n_timed'] = int(m.group(4))
            rec['total_time_us_std'] = float(m.group(5)) * 1000
        recs.append(rec)
    return recs


def parse_vamana(log: str, text: str, nq: int) -> List[dict]:
    """Rows of the search_disk_index table: 9 columns, or 16 with the counters of the instrumented build.

    run_vamana.sh repeats L; the first line of its log gives the number of warm-up rows per L, which are dropped,
    and the other rows of an L are averaged. A log without that line gives one record per row."""
    m = re.search(r'^### runs per L: warmup (\d+), timed (\d+)', text, flags=re.M)
    n_warmup = int(m.group(1)) if m else 0
    rows: Dict[str, List[List[float]]] = {}
    recs = []
    for line in text.splitlines():
        tok = line.split()
        if len(tok) not in (9, 16) or not all(re.fullmatch(NUM, t) for t in tok):
            continue
        v = [float(t) for t in tok]
        if not m:
            rec = {'param': 'L=' + tok[0], 'qps': v[2], 'recall': v[-1] / 100.0, 'total_time_us': nq / v[2] * 1e6}
            if len(v) == 16:
                rec['path_total'] = v[13] * nq  # mean per query
                rec['calc_total'] = v[14] * nq
            recs.append(rec)
            continue
        rows.setdefault(tok[0], []).append(v)
    for L, vs in rows.items():
        timed = vs[n_warmup:]
        if not timed:
            continue
        times = [nq / v[2] * 1e6 for v in timed]
        rec = {'param': 'L=' + L, 'qps': nq / mean(times) * 1e6, 'recall': mean([v[-1] for v in timed]) / 100.0,
               'total_time_us': mean(times), 'total_time_us_std': stdev(times), 'n_timed': len(timed)}
        if len(timed[0]) == 16:
            rec['path_total'] = mean([v[13] for v in timed]) * nq  # mean per query
            rec['calc_total'] = mean([v[14] for v in timed]) * nq
        recs.append(rec)
    return recs


PARSERS = {'song': parse_song, 'ggnn': parse_ggnn, 'ganns': parse_ganns, 'cagra': parse_cagra,
           'bang': parse_bang, 'pilotann': parse_pilotann, 'vamana': parse_vamana}


def read_meta(path: str) -> Dict[str, str]:
    with open(path) as f:
        return dict(line.rstrip('\n').split('\t', 1) for line in f if '\t' in line)


def collect(root: str) -> List[dict]:
    """All records under root, one per measured search setting."""
    out = []
    for dirpath, _, files in sorted(os.walk(root)):
        if 'meta.tsv' not in files:
            continue
        meta = read_meta(os.path.join(dirpath, 'meta.tsv'))
        system, nq = meta['system'], int(meta['nq'])
        run = os.path.relpath(os.path.dirname(dirpath), os.path.join(root, system))
        for name in sorted(files):
            if not name.endswith('.log'):
                continue
            log = os.path.join(dirpath, name)
            text = open(log, errors='replace').read()
            sections = timed_sections(text)
            if sections is None:
                recs = PARSERS[system](log, text, nq)
            else:
                recs = merge_runs([PARSERS[system](log, t, nq) for t in sections]) if sections else []
            for rec in recs:
                rec.setdefault('param', os.path.splitext(name)[0])
                rec.update(system=SYSTEM[system], dataset=meta['dataset'], run=run, k=int(meta['k']), nq=nq,
                           source=os.path.relpath(log, root))
                out.append(rec)
    return out


def pick(recs: List[dict], target: float) -> List[dict]:
    """Per (system, run, k): the record whose recall is closest to target."""
    best: Dict[tuple, dict] = {}
    for r in recs:
        if r.get('recall') is None:
            continue
        key = (r['system'], r['run'], r['k'])
        if key not in best or abs(r['recall'] - target) < abs(best[key]['recall'] - target):
            best[key] = r
    return [best[k] for k in sorted(best)]


def fmt(v: object) -> object:
    if v is None:
        return ''
    if isinstance(v, float):
        return f'{v:.10g}'
    return v


def write(rows: List[dict], cols: List[str], out: Optional[str]) -> None:
    f = open(out, 'w', newline='') if out else sys.stdout
    w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow({c: fmt(r.get(c)) for c in cols})
    if out:
        f.close()


BUILD_PATTERNS = {
    'ct_s': [(r'^Build time: (\d+)ms', 1e-3), (r'^Build time: ' + NUM + ' s', 1.0),
             (r'Graph built in ' + NUM + ' ms', 1e-3),
             (r'^Running time: ' + NUM + ' seconds', 1.0), (r'CAGRA index built in (\d+) ms', 1e-3),
             (r'Indexing time: ' + NUM, 1.0), (r'Build completed in ' + NUM + ' seconds', 1.0)],
    'avg_degree': [(r'^Average [Dd]egree: ' + NUM, 1.0)],
    'components': [(r'^Total number of connected components: (\d+)', 1.0),
                   (r'^Number of [Cc]onnected [Cc]omponents: (\d+)', 1.0)],
    'peak_host_gb': [(r'Maximum resident set size \(kbytes\): (\d+)', 1.0 / 1024 ** 2)],
}


def parse_build(root: str) -> List[dict]:
    rows = []
    base = os.path.join(root, 'build')
    for system in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        for name in sorted(os.listdir(os.path.join(base, system))):
            if not name.endswith('.log'):
                continue
            text = open(os.path.join(base, system, name), errors='replace').read()
            run = name[:-4]
            row = {'system': SYSTEM.get(system, system), 'run': run, 'source': os.path.join('build', system, name)}
            for col, pats in BUILD_PATTERNS.items():
                for pat, scale in pats:
                    v = first(pat, text, last=(col == 'peak_host_gb'))
                    if v is not None:
                        row[col] = round(v * scale, 3)
                        break
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('what', choices=['curves', 'stages', 'iterations', 'build'])
    parser.add_argument('--root', default=os.path.join(os.environ.get('OUT_DIR', ''), 'original'),
                        help='output root of the run scripts (default $OUT_DIR/original)')
    parser.add_argument('--out', default=None, help='CSV file (default: stdout)')
    parser.add_argument('--target', type=float, default=0.9, help='recall of the operating point')
    parser.add_argument('--k', type=int, default=10, help='K of Recall@K for stages and iterations')
    parser.add_argument('--all-runs', action='store_true', help='stages and iterations of every run')
    args = parser.parse_args()

    if args.what == 'build':
        write(parse_build(args.root), ['system', 'run', 'ct_s', 'avg_degree', 'components', 'peak_host_gb', 'source'],
              args.out)
        return
    recs = collect(args.root)
    if args.what == 'curves':
        write(recs, ['run', 'dataset', 'system', 'k', 'param', 'recall', 'qps', 'source', 'total_time_us',
                     'total_time_us_std', 'n_timed'], args.out)
        return
    if not args.all_runs:
        recs = [r for r in recs if r['run'] == r['dataset'] and r['k'] == args.k]
    if args.what == 'stages':
        rows = []
        for r in pick([r for r in recs if 'stages' in r], args.target):
            rows.append({**r, **r['stages']})
        write(rows, ['dataset', 'system', 'param', 'recall'] + STAGES + ['total_time_us', 'run', 'source',
                                                                         'total_time_us_std', 'n_timed'], args.out)
    else:
        rows = []
        for r in pick([r for r in recs if r.get('path_total') is not None], args.target):
            rows.append({'path_from': 'counter', **r, 'panel': 'a',
                         'dataset': NAME_FIG05.get(r['dataset'], r['dataset']),
                         'hops_per_query': r['path_total'] / r['nq']})
        write(rows, ['panel', 'dataset', 'system', 'param', 'recall', 'path_total', 'nq', 'hops_per_query',
                     'path_from', 'run', 'source', 'n_timed'], args.out)


if __name__ == '__main__':
    main()
