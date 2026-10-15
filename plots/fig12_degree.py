"""Figure 12: QPS versus Recall@10 on SIFT1M with maximum degree 64 and 128.

Input: results/fig12_degree.csv. The panels keep the axis ranges of the original manuscript; the y range grows
when the PilotANN curve needs more room. Output: fig12_degree<R>.pdf and fig12_legend.pdf.
"""
import logging
import math
import os

import matplotlib.pyplot as plt

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SYSTEMS = ['BANG', 'CAGRA', 'Vamana', 'GANNS', 'GGNN', 'HNSW_1', 'PilotANN', 'SONG']
LATE = ['PilotANN']  # drawn last, and the y range grows to fit it
LIMITS = {64: ((0.6632, 1.0), (5.647019, 2.07164e+06)), 128: ((0.8064, 1.0), (23.385728, 1.27326e+06))}


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig12_degree.csv')
    for degree, ((x_lo, x_hi), (lo, hi)) in LIMITS.items():
        sub = d[d['degree'] == degree]
        curves = [(s, list(g['recall']), list(g['qps'])) for s in SYSTEMS
                  for g in [sub[sub['system'] == s]] if len(g)]
        extra = [c for c in curves if c[0] in LATE]
        curves = [c for c in curves if c[0] not in LATE]
        for _, xs, ys in extra:
            inside = [y for x, y in zip(xs, ys) if x_lo <= x <= x_hi]
            if inside:
                lo, hi = min(lo, min(inside) / 1.2), max(hi, max(inside) * 1.2)
        fig, ax = plt.subplots(figsize=(8, 6))
        style.draw_curves(ax, curves + extra, (x_lo, x_hi), (math.log10(lo), math.log10(hi)), 8, 1.3)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(lo, hi)
        style.finish_qps_panel(fig, ax, 'Recall@K', 0.47 * style.COL_W, os.path.join(args.out, f'fig12_degree{degree}.pdf'))
    style.system_legend(SYSTEMS, os.path.join(args.out, 'fig12_legend.pdf'), ncol=4)
    logger.info('wrote fig12_*.pdf')


if __name__ == '__main__':
    main()
