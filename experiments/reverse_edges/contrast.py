"""Figure 16, step 4: tables that compare the graphs built with and without reverse edges.

Reads, under the experiment directory (default $OUT_DIR/reverse_edges):
  edge_stats/summary.tsv                    short and long edges of both graphs (edge_stats.sh)
  search/{with_reverse,no_reverse}/sweep.tsv  Recall@10 and QPS per iteration count (search.sh)
Writes:
  contrast_edge.tsv    one row per graph plus the difference no_reverse - with_reverse (panel b)
  contrast.tsv         Recall@10, QPS and kernel time of both graphs at every iteration count (panel a)
  contrast_summary.txt short text summary of both tables
"""
import argparse
import csv
import os
from pathlib import Path

GRAPHS = ('with_reverse', 'no_reverse')


def load_tsv(path: Path) -> list:
    """Rows of a tab-separated file with a header line, as dicts."""
    with open(path) as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_edge_table(edge_by: dict, path: Path) -> None:
    """Short and long edge shares and edge lengths of both graphs, and their difference."""
    with path.open('w') as f:
        f.write('graph\tn_sample\tK\tn_edges\tn_short\tn_long\tshort_ratio\tlong_ratio\t'
                'mean_len_all\tmean_len_short\tmean_len_long\tp50_len_all\tp90_len_all\n')
        for key in GRAPHS:
            r = edge_by[key]
            short_r = float(r['short_ratio'])
            f.write(
                f"{key}\t{r['n_sample']}\t{r['K']}\t{r['n_edges']}\t{r['n_short']}\t{r['n_long']}\t"
                f"{short_r:.6f}\t{1.0 - short_r:.6f}\t"
                f"{r['mean_len_all']}\t{r['mean_len_short']}\t{r['mean_len_long']}\t"
                f"{r['p50_len_all']}\t{r['p90_len_all']}\n"
            )
        a, b = edge_by['with_reverse'], edge_by['no_reverse']
        sa, sb = float(a['short_ratio']), float(b['short_ratio'])
        f.write(
            f"delta(no_reverse-with_reverse)\t\t\t\t\t\t{sb - sa:+.6f}\t{(1 - sb) - (1 - sa):+.6f}\t"
            f"{float(b['mean_len_all']) - float(a['mean_len_all']):+.6f}\t"
            f"{float(b['mean_len_short']) - float(a['mean_len_short']):+.6f}\t"
            f"{float(b['mean_len_long']) - float(a['mean_len_long']):+.6f}\t"
            f"{float(b['p50_len_all']) - float(a['p50_len_all']):+.6f}\t"
            f"{float(b['p90_len_all']) - float(a['p90_len_all']):+.6f}\n"
        )


def write_contrast(edge_by: dict, wr: dict, nr_list: list, path: Path) -> None:
    """Both sweeps side by side, matched on (it, itopk), with the differences no_reverse - with_reverse."""
    sr_wr = float(edge_by['with_reverse']['short_ratio'])
    sr_nr = float(edge_by['no_reverse']['short_ratio'])
    with path.open('w') as f:
        f.write('# sift1M CAGRA reverse A/B contrast\n')
        f.write('# graphs: shared nn-descent (gd=32,igd=64,niter=20); with_reverse vs no_reverse optimize\n')
        f.write('# edge sample: sample/ (S=10000,K=32); see contrast_edge.tsv\n')
        f.write('# search: same it_schedule / itopk=max(32,ceil32(it)) / bs=10000 / k=10 / search_width=1\n')
        f.write('# short_ratio with_reverse=%.6f  no_reverse=%.6f  delta=%+.6f\n' % (sr_wr, sr_nr, sr_nr - sr_wr))
        f.write(
            'it\titopk\tnq\t'
            'recall@10_with_reverse\tqps_with_reverse\tkernel_ms_with_reverse\t'
            'recall@10_no_reverse\tqps_no_reverse\tkernel_ms_no_reverse\t'
            'd_recall@10\td_qps\td_qps_frac\td_kernel_ms\n'
        )
        for r in nr_list:
            a = wr[(r['it'], r['itopk'])]
            ra, rb = float(a['recall@10']), float(r['recall@10'])
            qa, qb = float(a['qps']), float(r['qps'])
            ka, kb = float(a['kernel_ms']), float(r['kernel_ms'])
            f.write(
                f"{r['it']}\t{r['itopk']}\t{r['nq']}\t"
                f"{ra:.6f}\t{qa:.2f}\t{ka:.4f}\t"
                f"{rb:.6f}\t{qb:.2f}\t{kb:.4f}\t"
                f"{rb - ra:+.6f}\t{qb - qa:+.2f}\t{(qb - qa) / qa:+.6f}\t{kb - ka:+.4f}\n"
            )


def write_summary(edge_by: dict, wr: dict, nr_list: list, path: Path) -> None:
    """Edge shares and Recall@10 / QPS at a few iteration counts, as aligned text."""
    with path.open('w') as f:
        f.write('sift1M CAGRA reverse A/B contrast\n')
        f.write('graphs: shared nn-descent; optimize with_reverse vs no_reverse\n')
        f.write('edge sample: sample/ S=10000 K=32\n')
        f.write('search: it_schedule + itopk=max(32,ceil32(it)) bs=10000 k=10\n\n')
        f.write('== edge short/long ==\n')
        f.write(f"{'graph':<14} {'short_ratio':>12} {'long_ratio':>12} {'mean_len':>12} {'n_short':>10} {'n_long':>10}\n")
        for key in GRAPHS:
            r = edge_by[key]
            sr = float(r['short_ratio'])
            f.write(f"{key:<14} {sr:12.6f} {1 - sr:12.6f} {float(r['mean_len_all']):12.3f} "
                    f"{r['n_short']:>10} {r['n_long']:>10}\n")
        delta = float(edge_by['no_reverse']['short_ratio']) - float(edge_by['with_reverse']['short_ratio'])
        f.write(f"{'delta':<14} {delta:+12.6f}\n")
        f.write('\n== recall@10 / qps (same it, itopk) ==\n')
        f.write(f"{'it':>5} {'itopk':>5} {'rec_wr':>10} {'rec_nr':>10} {'d_rec':>9} "
                f"{'qps_wr':>12} {'qps_nr':>12} {'d_qps%':>9}\n")
        picks = {15, 20, 32, 48, 64, 96, 128, 192, 256, 500}
        for r in nr_list:
            if int(r['it']) not in picks:
                continue
            a = wr[(r['it'], r['itopk'])]
            ra, rb = float(a['recall@10']), float(r['recall@10'])
            qa, qb = float(a['qps']), float(r['qps'])
            f.write(f"{r['it']:>5} {r['itopk']:>5} {ra:10.6f} {rb:10.6f} {rb - ra:+9.6f} "
                    f"{qa:12.0f} {qb:12.0f} {(qb - qa) / qa * 100:+8.2f}%\n")
        f.write('\nfull tables: contrast.tsv (all its), contrast_edge.tsv\n')


def main() -> None:
    default = os.path.join(os.environ['OUT_DIR'], 'reverse_edges') if os.environ.get('OUT_DIR') else None
    ap = argparse.ArgumentParser(description='Contrast tables of Figure 16.')
    ap.add_argument('--dir', default=default, required=default is None,
                    help='experiment directory (default: $OUT_DIR/reverse_edges)')
    args = ap.parse_args()
    out = Path(args.dir)

    edge_by = {r['algo']: r for r in load_tsv(out / 'edge_stats' / 'summary.tsv')}
    wr = {(r['it'], r['itopk']): r for r in load_tsv(out / 'search' / 'with_reverse' / 'sweep.tsv')}
    nr_list = load_tsv(out / 'search' / 'no_reverse' / 'sweep.tsv')

    write_edge_table(edge_by, out / 'contrast_edge.tsv')
    write_contrast(edge_by, wr, nr_list, out / 'contrast.tsv')
    write_summary(edge_by, wr, nr_list, out / 'contrast_summary.txt')
    for name in ('contrast_edge.tsv', 'contrast.tsv', 'contrast_summary.txt'):
        print('wrote', out / name)


if __name__ == '__main__':
    main()
