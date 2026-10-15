"""Combine the Nsight Compute results <root>/<dataset>_<system>[_fp16]/ncu/raw.csv into one row per group.

Columns: dataset, system, precision, kernel name, launches, iterations, Recall@10, SM, GPU DRAM, Occ., Thr., L2,
main stall reason and its share, share of math_pipe_throttle, GPU DRAM bytes read per query
(dram__bytes_read.sum divided by the number of queries), Nsight Compute version, GPU, SM clock during the
measurement. raw.csv is exported by profile_ncu.sh with --print-units base, so bytes are bytes.

Stall share = smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio of the reason divided by the sum
over all stall reasons. selected (issuing an instruction) and not_selected (ready but not picked) are not stalls
and are left out of the sum, as is not_issued. math_pipe_share_pct is the share of math_pipe_throttle (waiting
for an arithmetic pipeline) with the same denominator; it shows whether arithmetic limits the kernel.

When raw.csv has several rows (BANG launches its main kernel once per iteration, and all launches of one measured
search are profiled), they are combined into one row. Each ratio is weighted by its own denominator, which equals
summing numerators and denominators over the launches: SM and GPU DRAM throughput by duration
(gpu__time_duration.sum), Occ. by active SM cycles (sm__cycles_active.sum), Thr. by instructions
(smsp__inst_executed.sum), the stall ratios by issue-active cycles (smsp__issue_active.sum). The L2 hit rate is
weighted by L2 sectors (lts__t_sectors.sum), which is close to but not the same as its denominator (sectors
looked up). GPU DRAM bytes are summed.

usage: python summarize.py <profiling dir, $OUT_DIR/profiling> [--out summary.tsv]
"""
import argparse
import csv
import re
from pathlib import Path

COLS = {
    'SM': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'GPU DRAM': 'dram__throughput.avg.pct_of_peak_sustained_elapsed',
    'Occ.': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'Thr.': 'smsp__thread_inst_executed_per_inst_executed.ratio',
    'L2': 'lts__t_sector_hit_rate.pct',
}
WEIGHT = {
    'SM': 'gpu__time_duration.sum',
    'GPU DRAM': 'gpu__time_duration.sum',
    'Occ.': 'sm__cycles_active.sum',
    'Thr.': 'smsp__inst_executed.sum',
    'L2': 'lts__t_sectors.sum',
}
STALL_WEIGHT = 'smsp__issue_active.sum'
STALL = re.compile(r'^smsp__average_warps_issue_stalled_(\w+)_per_issue_active\.ratio$')


def read_raw(path: Path) -> list:
    """Return one dict per launch (the row after the header holds the units)."""
    with open(path) as f:
        rows = list(csv.reader(f))
    head = next(i for i, r in enumerate(rows) if 'Kernel Name' in r)
    return [dict(zip(rows[head], r)) for r in rows[head + 2:] if r]


def num(s: str) -> float:
    return float(s.replace(',', ''))


def wavg(launches: list, key: str, weight: str) -> float:
    """Weighted mean over the launches; the value itself for a single launch."""
    if len(launches) == 1:
        return num(launches[0][key])
    ws = [num(l[weight]) for l in launches]
    return sum(num(l[key]) * w for l, w in zip(launches, ws)) / sum(ws)


def combine(launches: list) -> dict:
    """Combine the launches into one row: the COLS columns, the stall ratios, total GPU DRAM bytes read, launches."""
    first = launches[0]
    out = {c: wavg(launches, k, WEIGHT[c]) for c, k in COLS.items() if k in first}
    out['stalls'] = {STALL.match(k).group(1): wavg(launches, k, STALL_WEIGHT)
                     for k, v in first.items() if STALL.match(k) and v}
    if first.get('dram__bytes_read.sum'):
        out['dram_bytes'] = sum(num(l['dram__bytes_read.sum']) for l in launches)
    out['kernel'] = first.get('Kernel Name', '')
    out['launches'] = len(launches)
    # SM clock during the measurement in MHz, weighted by duration; empty if the metric is missing
    clk = 'sm__cycles_elapsed.avg.per_second'
    out['sm_clock_mhz'] = wavg(launches, clk, 'gpu__time_duration.sum') / 1e6 if first.get(clk) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('root', type=Path, help='profiling output directory ($OUT_DIR/profiling)')
    ap.add_argument('--out', type=Path, default=None, help='output table (default <root>/summary.tsv)')
    args = ap.parse_args()
    out = []
    for raw in sorted(args.root.glob('*/ncu/raw.csv')):
        tag = raw.parent.parent.name
        prec = 'fp16' if tag.endswith('_fp16') else 'fp32'
        ds, sysn = tag.removesuffix('_fp16').split('_', 1)
        m = combine(read_raw(raw))
        stalls = m['stalls']
        for k in ('not_issued', 'selected', 'not_selected'):  # states that are not stalls
            stalls.pop(k, None)
        top = max(stalls, key=stalls.get) if stalls else ''
        with open(raw.parent / 'run' / 'sweep.tsv') as f:
            sweep = list(csv.DictReader(f, delimiter='\t'))[-1]
        nq = int(sweep.get('nq', 0) or 0)
        bytes_q = m['dram_bytes'] / nq if nq and 'dram_bytes' in m else None
        out.append({'dataset': ds, 'system': sysn, 'precision': prec, 'kernel': m['kernel'],
                    'launches': m['launches'],
                    'it': sweep.get('it', ''),
                    'recall@10': sweep.get('recall@10', ''),
                    **{c: f"{m[c]:.1f}" for c in COLS if c in m},
                    'stall': top, 'stall_share_pct': f"{100 * stalls[top] / sum(stalls.values()):.0f}" if top else '',
                    'math_pipe_share_pct': f"{100 * stalls.get('math_pipe_throttle', 0) / sum(stalls.values()):.1f}" if stalls else '',
                    'dram_read_bytes_per_query': f'{bytes_q:.0f}' if bytes_q is not None else '',
                    'ncu': (raw.parent / 'ncu_version.txt').read_text().strip(),
                    'gpu': (raw.parent / 'gpu.txt').read_text().strip(),
                    'sm_clock_mhz': f"{m['sm_clock_mhz']:.0f}" if m['sm_clock_mhz'] else ''})
    if not out:
        raise SystemExit(f'no */ncu/raw.csv under {args.root}')
    path = args.out or args.root / 'summary.tsv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(out)
    print(f'wrote {path} ({len(out)} rows)')


if __name__ == '__main__':
    main()
