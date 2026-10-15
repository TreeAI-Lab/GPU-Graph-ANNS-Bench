"""Figure 10: QPS at 90% Recall@10 on DEEP1M, DEEP10M and DEEP100M. A triangle marks a build that ran out of memory.

Input: results/fig10_scalability.csv (kernel time per 10,000 queries). Output: fig10_scalability.pdf.
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATASETS = [('deep1M', 'DEEP1M'), ('deep10M', 'DEEP10M'), ('deep100M', 'DEEP100M')]
COLORS = [(244, 241, 222), (223, 122, 94), (60, 64, 91)]
BAR_W = 0.26
OOM = dict(marker='v', markersize=18, color='yellow', markeredgecolor='black', markeredgewidth=1)


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig10_scalability.csv')
    d['qps'] = (1e4 / (d['time_us_per_10k'] / 1e6)).fillna(0)
    systems = list(d['system'].unique())
    x_idx = range(len(systems))
    plt.figure(figsize=(8, 6))
    for i, (ds, name) in enumerate(DATASETS):
        sub = d[d['dataset'] == ds]
        bars = plt.bar([x + i * BAR_W for x in x_idx], sub['qps'], width=BAR_W,
                       color=tuple(c / 255 for c in COLORS[i]), edgecolor='black', linewidth=2.25, label=name)
        for bar, qps in zip(bars, sub['qps']):
            if qps == 0:  # out of memory: arrow from 1e4 down and a triangle above it
                xp = bar.get_x() + bar.get_width() / 2
                plt.arrow(xp, 1e4, 0, -9.8e3, head_width=0.05, head_length=50, fc='red', ec='red', linewidth=4)
                plt.plot(xp, 2e4, **OOM)
    plt.ylabel('QPS', fontdict={'family': 'Times New Roman'})
    plt.yscale('log')
    plt.ylim(1e2, 1e8)
    plt.yticks(fontproperties='Times New Roman')
    plt.xticks([x + BAR_W * (len(DATASETS) - 1) / 2 for x in x_idx], systems, fontproperties='Times New Roman',
               rotation=45)
    h, labels = plt.gca().get_legend_handles_labels()  # two rows above the axes
    order = style.row_major(list(range(len(h))), 2)
    style.axes_legend(plt.gca(), [h[i] for i in order], [labels[i] for i in order],
                      prop={'family': 'Times New Roman'}, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
                      frameon=False, handlelength=1.2, handletextpad=0.4, columnspacing=1.2)
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
    out = os.path.join(args.out, 'fig10_scalability.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
