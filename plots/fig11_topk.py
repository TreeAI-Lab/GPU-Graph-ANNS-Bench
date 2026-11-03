"""Figure 11: QPS versus Recall@K on SIFT1M for K = 10, 40, 70 and 100.

Input: results/fig11_topk.csv. Every panel plots Recall@K from 0.6 to the highest recall of a GPU system.
Output: fig11_top<K>.pdf and fig11_legend.pdf.
"""
import logging
import math
import os

import matplotlib.pyplot as plt

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

HNSW = ['HNSW_1', 'HNSW_10', 'HNSW_20', 'HNSW_40']
WIDTH = 0.23 * style.TEXT_W


def curves_of(d, systems: list) -> list:
    return [(s, list(d[d['system'] == s]['recall']), list(d[d['system'] == s]['qps']))
            for s in systems if (d['system'] == s).any()]


def panel(d, k: int, out: str) -> None:
    d = d[d['recall'] >= 0.6]
    curves = curves_of(d, [s for s in style.SYSTEMS if s not in HNSW]) + curves_of(d, HNSW)
    x_range = (0.6, d[~d['system'].isin(HNSW)]['recall'].max())  # the axis ends at the highest GPU-system recall
    all_y = [y for _, xs, ys in curves for x, y in zip(xs, ys) if x <= x_range[1]]
    fig, ax = plt.subplots(figsize=(8, 6))
    style.draw_curves(ax, curves, x_range, (math.log10(min(all_y)), math.log10(max(all_y))), 11, 1.5)
    ax.set_xlim(*x_range)
    style.finish_qps_panel(fig, ax, f'Recall@{k}', WIDTH, out)


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig11_topk.csv')
    for k in (10, 40, 70, 100):
        panel(d[d['k'] == k], k, os.path.join(args.out, f'fig11_top{k}.pdf'))
    style.system_legend(style.SYSTEMS, os.path.join(args.out, 'fig11_legend.pdf'), ncol=6, width_pt=style.TEXT_W)
    logger.info('wrote fig11_*.pdf')


if __name__ == '__main__':
    main()
