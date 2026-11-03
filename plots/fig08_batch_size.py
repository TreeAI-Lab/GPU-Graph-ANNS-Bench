"""Figure 8: QPS and time per batch on SIFT1M at 95% Recall@10 for batch sizes 1 to 10,000.

Batches run one after another, so the time per batch is the batch size divided by QPS.
Input: results/fig08_batch_size.csv. Output: fig08_qps.pdf, fig08_latency.pdf, fig08_legend.pdf.
"""
import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

COLORS = [tuple(c / 255 for c in rgb) for rgb in
          [(244, 241, 222), (223, 122, 94), (60, 64, 91), (139, 178, 154), (242, 204, 142)]]
WIDTH = 0.17


def panel(d: pd.DataFrame, systems: List[str], batches: List[int], column: str, ylabel: str, out: str) -> None:
    x = np.arange(len(systems))
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, b in enumerate(batches):
        vals = [d[(d['system'] == s) & (d['batch_size'] == b)][column].values[0] for s in systems]
        ax.bar(x + i * WIDTH, vals, WIDTH, label=f'Batch {b}', color=COLORS[i % len(COLORS)], edgecolor='black',
               linewidth=2.25)
    ax.set_xticks(x + WIDTH * (len(batches) - 1) / 2)
    ax.set_xticklabels(systems)
    plt.xticks(fontproperties='Times New Roman', rotation=60, ha='right', rotation_mode='anchor')
    plt.yticks(fontproperties='Times New Roman')
    plt.ylabel(ylabel, fontdict={'family': 'Times New Roman'})
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_yscale('log')
    style.fit_text(fig, 0.47 * style.COL_W, tight_layout=True)  # two panels side by side in one column
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info('wrote %s', out)


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig08_batch_size.csv')
    d['batch_ms'] = d['batch_size'] / d['qps'] * 1000.0
    systems, batches = list(d['system'].unique()), sorted(d['batch_size'].unique())
    panel(d, systems, batches, 'qps', 'QPS', os.path.join(args.out, 'fig08_qps.pdf'))
    panel(d, systems, batches, 'batch_ms', 'Latency (ms)', os.path.join(args.out, 'fig08_latency.pdf'))
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[i % len(COLORS)], edgecolor='black', linewidth=0.8,
                             label=f'Batch {b}') for i, b in enumerate(batches)]
    out = os.path.join(args.out, 'fig08_legend.pdf')
    style.save_legend(handles, out, ncol=3, prop={'family': 'Times New Roman', 'size': 9}, columnspacing=1.0,
                      handlelength=1.2, handletextpad=0.4)
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
