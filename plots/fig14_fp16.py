"""Figure 14: QPS versus Recall@10 on the A10 with FP32 (solid, hollow markers) and FP16 (dashed, filled markers).

Input: results/fig14_fp16.csv (mean of three sweeps). Points are kept at least MIN_GAP apart on the plot so that the
solid and dashed lines stay apart. Output: fig14_<dataset>.pdf and fig14_legend.pdf.
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SYSTEMS = ['SONG', 'GGNN', 'GANNS', 'CAGRA']
LINE = {'fp32': '-', 'fp16': '--'}
MIN_RECALL = 0.6
MIN_GAP = 0.1  # minimum distance between kept points in normalized axes coordinates


def thin(d: pd.DataFrame, x_range: tuple, log_y_range: tuple) -> pd.DataFrame:
    """Keep points at least MIN_GAP apart along the curve (d sorted by recall); the last point is always kept."""
    xs = (d['recall_at_10'].to_numpy() - x_range[0]) / (x_range[1] - x_range[0])
    ys = (np.log10(d['qps'].to_numpy()) - log_y_range[0]) / (log_y_range[1] - log_y_range[0])
    keep, last = [0], 0
    for i in range(1, len(d)):
        if np.hypot(xs[i] - xs[last], ys[i] - ys[last]) >= MIN_GAP:
            keep.append(i)
            last = i
    if keep[-1] != len(d) - 1:
        keep.append(len(d) - 1)
    return d.iloc[keep]


def panel(d: pd.DataFrame, out: str) -> None:
    curves = {(s, p): d[(d['system'] == s) & (d['precision'] == p)].sort_values('recall_at_10')
              for s in SYSTEMS for p in LINE}
    curves = {k: c[c['recall_at_10'] >= MIN_RECALL] for k, c in curves.items()}
    x_right = max(c['recall_at_10'].max() for c in curves.values())
    qps = pd.concat([c['qps'] for c in curves.values()])
    log_y_range = (np.log10(qps.min()), np.log10(qps.max()))
    plt.figure(figsize=(8, 6))
    for s in SYSTEMS:
        for p, ls in LINE.items():
            c = thin(curves[(s, p)], (MIN_RECALL, x_right), log_y_range)
            plt.plot(c['recall_at_10'], c['qps'], linestyle=ls, marker=style.MARKER[s], markersize=9,
                     markerfacecolor=style.COLOR[s] if p == 'fp16' else 'none', markeredgecolor=style.COLOR[s],
                     markeredgewidth=1.5, color=style.COLOR[s], alpha=0.95, linewidth=2)
    plt.yscale('log')
    plt.xlabel('Recall@10')
    plt.ylabel('QPS')
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(MIN_RECALL, x_right)
    style.fit_text(plt.gcf(), 0.23 * style.TEXT_W)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info('wrote %s', out)


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig14_fp16.csv')
    for ds in ['gist', 'deep1M']:
        panel(d[d['dataset'] == ds], os.path.join(args.out, f'fig14_{ds}.pdf'))
    handles = [Line2D([], [], color=style.COLOR[s], marker=style.MARKER[s], markersize=7, markerfacecolor='none',
                      markeredgewidth=1.5, linewidth=2, label=s) for s in SYSTEMS]
    handles += [Line2D([], [], color='black', linestyle='-', linewidth=2, label='FP32'),
                Line2D([], [], color='black', linestyle='--', linewidth=2, label='FP16')]
    style.save_legend(handles, os.path.join(args.out, 'fig14_legend.pdf'), ncol=3,
                      prop={'family': 'Times New Roman', 'size': 9})


if __name__ == '__main__':
    main()
