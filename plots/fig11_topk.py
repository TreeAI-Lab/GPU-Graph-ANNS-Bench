"""Figure 11: QPS versus Recall@K on SIFT1M for K = 10, 40, 70 and 100.

Input: results/fig11_topk.csv. The K = 40, 70, 100 panels keep the axis ranges of the original manuscript; the y range
grows when the HNSW curves need more room. Output: fig11_top<K>.pdf and fig11_legend.pdf.
"""
import logging
import math
import os

import matplotlib.pyplot as plt

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

HNSW = ['HNSW_1', 'HNSW_10', 'HNSW_20', 'HNSW_40']
# Axis ranges of the original-manuscript panels: (x min, x max), (y min, y max)
LIMITS = {40: ((0.603, 1.0), (5.073415, 1.9393e+06)),
          70: ((0.6907, 0.999923), (5.173734, 1.44998e+06)),
          100: ((0.851424, 0.9992), (5.178596, 1.26002e+06))}
WIDTH = 0.23 * style.TEXT_W


def curves_of(d, systems: list) -> list:
    return [(s, list(d[d['system'] == s]['recall']), list(d[d['system'] == s]['qps']))
            for s in systems if (d['system'] == s).any()]


def top10(d, out: str) -> None:
    d = d[d['recall'] >= 0.6]
    curves = curves_of(d, [s for s in style.SYSTEMS if s not in HNSW]) + curves_of(d, HNSW)
    x_range = (0.6, d[~d['system'].isin(HNSW)]['recall'].max())  # the axis ends at the highest GPU-system recall
    all_y = [y for _, _, ys in curves for y in ys]
    fig, ax = plt.subplots(figsize=(8, 6))
    style.draw_curves(ax, curves, x_range, (math.log10(min(all_y)), math.log10(max(all_y))), 11, 1.5)
    ax.set_xlim(*x_range)
    style.finish_qps_panel(fig, ax, 'Recall@10', WIDTH, out)


def topk(d, k: int, out: str) -> None:
    (x_lo, x_hi), (lo, hi) = LIMITS[k]
    kdd = [s for s in style.SYSTEMS if s not in HNSW]
    curves, extra = curves_of(d, kdd), curves_of(d, HNSW)
    for _, xs, ys in extra:
        inside = [y for x, y in zip(xs, ys) if x_lo <= x <= x_hi]
        if inside:
            lo, hi = min(lo, min(inside) / 1.2), max(hi, max(inside) * 1.2)
    fig, ax = plt.subplots(figsize=(8, 6))
    style.draw_curves(ax, curves + extra, (x_lo, x_hi), (math.log10(lo), math.log10(hi)), 8, 1.3)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(lo, hi)
    style.finish_qps_panel(fig, ax, 'Recall@K', WIDTH, out)


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig11_topk.csv')
    top10(d[d['k'] == 10], os.path.join(args.out, 'fig11_top10.pdf'))
    for k in (40, 70, 100):
        topk(d[d['k'] == k], k, os.path.join(args.out, f'fig11_top{k}.pdf'))
    style.system_legend(style.SYSTEMS, os.path.join(args.out, 'fig11_legend.pdf'), ncol=6, width_pt=style.TEXT_W)
    logger.info('wrote fig11_*.pdf')


if __name__ == '__main__':
    main()
