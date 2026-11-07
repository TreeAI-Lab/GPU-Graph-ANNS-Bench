"""Figure 11: QPS near 95% Recall@10 on DEEP1M, DEEP10M and DEEP100M. A triangle marks a build that ran out of memory.

Input: results/fig11_scalability.csv (time per 10,000 queries at the plotted QPS: kernel time for SONG, GGNN, GANNS and CAGRA,
wall time for BANG and PilotANN; status below_target marks a curve that stops
below the minimum recall of summarize.py, drawn as a label). Output: fig11_scalability.pdf and fig11_legend.pdf.
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATASETS = [('deep1M', 'DEEP1M'), ('deep10M', 'DEEP10M'), ('deep100M', 'DEEP100M')]
COLORS = [(244, 241, 222), (223, 122, 94), (60, 64, 91)]
BAR_W = 0.26
OOM = dict(marker='v', markersize=18, color='yellow', markeredgecolor='black', markeredgewidth=1)
TARGET = 0.95  # recall at which results/fig11_scalability.csv reads the curves


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig11_scalability.csv')
    d['qps'] = (1e4 / (d['time_us_per_10k'] / 1e6)).fillna(0)
    d['status'] = d['status'].fillna('') if 'status' in d else ''
    systems = list(d['system'].unique())
    x_idx = range(len(systems))
    plt.figure(figsize=(8, 6))
    for i, (ds, name) in enumerate(DATASETS):
        sub = d[d['dataset'] == ds]
        bars = plt.bar([x + i * BAR_W for x in x_idx], sub['qps'], width=BAR_W,
                       color=tuple(c / 255 for c in COLORS[i]), edgecolor='black', linewidth=2.25, label=name)
        for bar, qps, status in zip(bars, sub['qps'], sub['status']):
            xp = bar.get_x() + bar.get_width() / 2
            if qps == 0 and status == 'below_target':  # the curve exists but stops below the minimum recall
                plt.text(xp, 1.5e2, f'<{TARGET:.0%}', rotation=90, ha='center', va='bottom', color='red',
                         fontdict={'family': 'Times New Roman'})
            elif qps == 0:  # out of memory: arrow from 1e4 down and a triangle above it
                plt.arrow(xp, 1e4, 0, -9.8e3, head_width=0.05, head_length=50, fc='red', ec='red', linewidth=4)
                plt.plot(xp, 2e4, **OOM)
    plt.ylabel('QPS', fontdict={'family': 'Times New Roman'})
    plt.yscale('log')
    plt.ylim(1e2, 1e8)
    plt.yticks(fontproperties='Times New Roman')
    plt.xticks([x + BAR_W * (len(DATASETS) - 1) / 2 for x in x_idx], systems, fontproperties='Times New Roman',
               rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    ax = plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    tx = xlim[0] + (xlim[1] - xlim[0]) * 0.05
    ty = 10 ** (np.log10(ylim[1]) - (np.log10(ylim[1]) - np.log10(ylim[0])) * 0.08)
    plt.plot(tx, ty, **OOM)
    plt.text(tx + (xlim[1] - xlim[0]) * 0.04, ty, 'means OOM.', color='red', fontdict={'family': 'Times New Roman'},
             verticalalignment='center')
    style.fit_text(plt.gcf(), 0.78 * style.COL_W, tight_layout=True)
    out = os.path.join(args.out, 'fig11_scalability.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info('wrote %s', out)
    handles = [Patch(facecolor=tuple(c / 255 for c in color), edgecolor='black', linewidth=0.8, label=name)
               for (_, name), color in zip(DATASETS, COLORS)]
    out = os.path.join(args.out, 'fig11_legend.pdf')
    style.save_legend(handles, out, ncol=len(handles), prop={'family': 'Times New Roman', 'size': 9},
                      handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
