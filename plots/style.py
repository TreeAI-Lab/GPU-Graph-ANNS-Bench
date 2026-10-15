"""Shared plotting helpers: font sizing for the paper layout, system colors, and curve thinning.

Figures are drawn at a large canvas size and scaled down by LaTeX. fit_text() sets the font sizes so that, once the
PDF is scaled to the width it has in the paper, axis labels and legends are 9 pt and tick labels 8 pt.
"""
import argparse
import logging
import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEXT_W = 495.0      # \textwidth of the journal template, pt
COL_W = 238.96      # \columnwidth, pt
TARGET_PT = 9.0     # axis labels, legends, text in the plot
TICK_PT = 8.0       # tick labels

# One color and marker per system, shared by all QPS-recall figures. The order is also the drawing order.
SYSTEMS = ['BANG', 'CAGRA', 'Vamana', 'GANNS', 'GGNN', 'HNSW_1', 'HNSW_10', 'HNSW_20', 'HNSW_40', 'PilotANN', 'SONG']
COLORS = ['#8B2E24', '#E63946', '#FF6B35', '#FFD23F', '#A8D840', '#06FFA5', '#6366F1', '#0077B6', '#AED9E0',
          '#FF5A5F', '#9D4EDD']
MARKERS = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '<', 'H', '1']
COLOR = dict(zip(SYSTEMS, COLORS))
MARKER = dict(zip(SYSTEMS, MARKERS))

# Non-graph GPU baselines: dashed black and gray lines
BASELINES = {
    'cuvs_ivf_pq': dict(label='cuVS IVF-PQ', color='#000000', marker='x'),
    'faiss_ivf_pq': dict(label='Faiss IVF-PQ', color='#6C757D', marker='+'),
}
BRUTE_FORCE = dict(label='cuVS Brute Force', color='#000000', marker='*')

MARK_GAP = 0.05  # minimum distance between two markers of a curve, in normalized axes coordinates


def parse_args(description: str) -> argparse.Namespace:
    """Common command line: where the result CSVs are and where the PDFs go."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--results', default=os.path.join(ROOT, 'results'), help='directory with the result CSVs')
    p.add_argument('--out', default=os.path.join(ROOT, 'plots', 'out'), help='output directory for the PDFs')
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    return args


def read(args: argparse.Namespace, name: str) -> pd.DataFrame:
    """Read results/<name>. Rows with a non-empty `excluded` column are dropped (the reason stays in the file)."""
    d = pd.read_csv(os.path.join(args.results, name))
    if 'excluded' in d.columns:
        d = d[d['excluded'].fillna('') == '']
    return d


def pareto_mask(qps: List[float]) -> List[bool]:
    """For QPS given in ascending recall, keep a point only if it is faster than every point with higher recall."""
    keep, best = [], -1.0
    for q in reversed(qps):
        keep.append(q > best)
        best = max(best, q)
    return keep[::-1]


def mark_index(xs: List[float], ys: List[float], x_range: Tuple[float, float],
               log_y_range: Tuple[float, float]) -> List[int]:
    """Indices of the points that get a marker: at least MARK_GAP apart on the log-y plot; first and last always.

    The line itself still passes through every point.
    """
    if not xs:
        return []
    nx = [(x - x_range[0]) / (x_range[1] - x_range[0]) for x in xs]
    ny = [(math.log10(y) - log_y_range[0]) / (log_y_range[1] - log_y_range[0]) for y in ys]
    keep, last = [0], 0
    for i in range(1, len(xs)):
        if math.hypot(nx[i] - nx[last], ny[i] - ny[last]) >= MARK_GAP:
            keep.append(i)
            last = i
    if keep[-1] != len(xs) - 1:
        keep.append(len(xs) - 1)
    return keep


def draw_curves(ax, curves: list, x_range: Tuple[float, float], log_y_range: Tuple[float, float],
                markersize: float, edgewidth: float) -> None:
    """Draw (system, recalls, qps) curves: sort by recall, drop dominated points, thin the markers."""
    for system, xs, ys in curves:
        pts = sorted(zip(xs, ys))
        pts = [p for p, k in zip(pts, pareto_mask([q for _, q in pts])) if k]
        if not pts:
            continue
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        ax.plot(xs, ys, marker=MARKER[system], markersize=markersize,
                markevery=mark_index(xs, ys, x_range, log_y_range), markerfacecolor='none',
                markeredgecolor=COLOR[system], markeredgewidth=edgewidth, color=COLOR[system], alpha=0.95,
                linewidth=2)


def finish_qps_panel(fig: Figure, ax, xlabel: str, width_pt: float, out: str) -> None:
    """Log QPS axis, Times tick labels, grid, font sizes for width_pt, save."""
    from matplotlib.ticker import MultipleLocator
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('QPS')
    lo, hi = ax.get_xlim()
    ax.xaxis.set_major_locator(MultipleLocator(0.1 if hi - lo > 0.15 else 0.05))
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily('Times New Roman')
    ax.grid(True, linestyle='--', alpha=0.3)
    fit_text(fig, width_pt)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)


def system_legend(systems: List[str], out: str, ncol: int, width_pt: float = None) -> None:
    """Legend PDF with one entry per system, 9 pt, inserted unscaled."""
    handles = [plt.Line2D([0], [0], marker=MARKER[s], markersize=6, markerfacecolor='none', markeredgecolor=COLOR[s],
                          markeredgewidth=1, color=COLOR[s], linewidth=1.2, alpha=0.85, label=s) for s in systems]
    save_legend(handles, out, ncol=ncol, prop={'family': 'Times New Roman', 'size': 9}, width_pt=width_pt,
                columnspacing=0.8, handletextpad=0.3, handlelength=1.6)


# ---------------------------------------------------------------- font sizing

def _tick_texts(fig: Figure) -> set:
    out = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            out.add(axis.get_offset_text())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                out.update((tick.label1, tick.label2))
    return out


def axes_legend(ax, *args, **kwargs):
    """Use instead of ax.legend: remembers the arguments so that fit_text can rebuild the legend at the new size.

    Legend spacing is fixed in points when the legend is created, so changing only the font size squeezes it.
    """
    leg = ax.legend(*args, **kwargs)
    leg._fit_args = (args, kwargs)
    return leg


def _rebuild_legends(fig: Figure, size: float) -> None:
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        if not hasattr(leg, '_fit_args'):
            logger.warning('legend not created by axes_legend; its spacing will not scale')
            continue
        args, kwargs = leg._fit_args
        kw = dict(kwargs)
        kw.pop('fontsize', None)
        prop = dict(kw.pop('prop', None) or {})
        prop['size'] = size
        ax.legend(*args, prop=prop, **kw)._fit_args = (args, kwargs)


def _set_size(fig: Figure, size: float, tick_size: float) -> None:
    _rebuild_legends(fig, size)
    for ax in fig.axes:
        ax.tick_params(which='both', labelsize=tick_size)
    ticks = _tick_texts(fig)
    for t in fig.findobj(Text):
        if t.get_text():
            t.set_fontsize(tick_size if t in ticks else size)


def fit_text(fig: Figure, width_pt: float, target_pt: float = TARGET_PT, tick_pt: float = TICK_PT,
             tight_layout: bool = False, rounds: int = 8) -> float:
    """Set font sizes so that the PDF saved with bbox_inches='tight', scaled to width_pt, has tick_pt ticks and
    target_pt text. Returns the scale factor. tight_layout=True re-runs tight_layout in every round."""
    pad_pt = plt.rcParams['savefig.pad_inches'] * 72
    k = k_prev = 1.0
    for _ in range(rounds):
        _set_size(fig, target_pt * k, tick_pt * k)
        if tight_layout:
            fig.tight_layout()
        fig.draw_without_rendering()
        bbox = fig.get_tightbbox()
        k_prev, k = k, (bbox.width * 72 + 2 * pad_pt) / width_pt
    if abs(k - k_prev) > 0.01 * k:
        logger.warning('font size did not converge (%.2f vs %.2f); some element is too wide at %gpt',
                       k_prev, k, target_pt)
    _set_size(fig, target_pt * k, tick_pt * k)
    if tight_layout:
        fig.tight_layout()
    _warn_clipped(fig)
    return k


def _warn_clipped(fig: Figure) -> None:
    """bbox_inches='tight' ignores the width of x labels and the height of y labels; warn when one is cut off."""
    fig.draw_without_rendering()
    box = fig.get_tightbbox()
    pad = plt.rcParams['savefig.pad_inches']
    to_in = fig.dpi_scale_trans.inverted()
    for ax in fig.axes:
        x = ax.xaxis.label.get_window_extent().transformed(to_in)
        y = ax.yaxis.label.get_window_extent().transformed(to_in)
        if ax.xaxis.label.get_text() and (x.x0 < box.x0 - pad or x.x1 > box.x1 + pad):
            logger.warning('x label %r is wider than the saved area', ax.xaxis.label.get_text())
        if ax.yaxis.label.get_text() and (y.y0 < box.y0 - pad or y.y1 > box.y1 + pad):
            logger.warning('y label %r is taller than the saved area', ax.yaxis.label.get_text())


def row_major(handles: list, ncol: int) -> list:
    """matplotlib fills legend columns first; reorder so that the legend reads row by row in the given order."""
    rows = -(-len(handles) // ncol)
    grid = [handles[i * ncol:(i + 1) * ncol] for i in range(rows)]
    return [row[c] for c in range(ncol) for row in grid if c < len(row)]


def save_legend(handles: list, out: str, ncol: int, size: float = TARGET_PT, width_pt: float = None,
                **kwargs) -> Tuple[float, float]:
    """Save a legend as its own PDF at its final size (insert it in LaTeX without scaling). Returns width, height.

    With width_pt the columns spread evenly over exactly that width (in TeX pt, 1/72.27 inch).
    """
    inch = (width_pt or 72.27) / 72.27
    fig = plt.figure(figsize=(inch, 1))
    if width_pt:
        kwargs.update(mode='expand', bbox_to_anchor=(0, 0, 1, 1), borderaxespad=0)
    leg = fig.legend(handles=row_major(handles, ncol), loc='center', ncol=ncol, frameon=False,
                     fontsize=size, **kwargs)
    fig.draw_without_rendering()
    bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if width_pt:
        bbox = Bbox.from_extents(0, bbox.y0 - 1 / 72, inch, bbox.y1 + 1 / 72)
    else:
        bbox = bbox.padded(1 / 72)
    fig.savefig(out, bbox_inches=bbox)
    plt.close(fig)
    return bbox.width * 72, bbox.height * 72
