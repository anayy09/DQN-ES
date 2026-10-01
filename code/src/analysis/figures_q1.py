"""
figures_q1.py — clean IEEE-style figure set for the DQN-ES manuscript.

Design choices (kept across every figure in the set):

* IEEE single-column width is 3.5 in; double-column is 7.16 in.  We use
  3.5 x 2.4 in by default, and 7.16 x 3.0 for two-panel figures.
* No plot titles.  LaTeX captions carry the same text, so duplicating
  inside the figure adds clutter.  The only exception is a panel label
  on multi-panel figures.
* Wong-style color-blind-safe palette.  On line plots, DQN-ES is the
    only filled marker with a thicker stroke; Pareto markers are filled
    for legibility.
* Shaded +/-1 SD bands instead of capped error bars on line plots; for
  bars we use thin capped error bars only.
* No legend inside the plot box when an alternative exists.  Either a
  single legend strip across the bottom or a one-line annotation next to
  the DQN-ES trace.
* No annotations on Pareto plots.  Markers + a small legend say the same
  thing without overlap.

Outputs all land in latex/figures/ as both .pdf (vector, for the
manuscript) and .png (raster preview).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.config import (
    EPSILON_DECAY,
    EPSILON_INIT,
    EPSILON_MIN,
    EPSILON_T_AT_0_05,
    EPSILON_T_AT_0_10,
    PRIMARY_SCALE,
)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':         8,
    'axes.labelsize':    8,
    'axes.titlesize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'legend.fontsize':   6.5,
    'legend.frameon':    False,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
    'axes.linewidth':    0.7,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.linestyle':    ':',
    'grid.linewidth':    0.4,
    'grid.alpha':        0.55,
    'lines.linewidth':   1.0,
    'lines.markersize':  4.5,
})

# ---------------------------------------------------------------------------
# Wong color-blind-safe palette (Nature Methods, Wong 2011)
# Bright orange (E69F00), sky blue (56B4E9), green (009E73), yellow (F0E442),
# vermillion (D55E00), reddish purple (CC79A7), blue (0072B2), black (000000)
# ---------------------------------------------------------------------------
COLORS: Dict[str, str] = {
    'DQN-ES':    '#D55E00',   # vermillion — primary method
    'PSO+DQN':   '#E69F00',   # bright orange — attribution control
    'ES-only':   '#CC79A7',   # reddish purple
    'DQN-only':  '#0072B2',   # blue
    'PSO':       '#009E73',   # green
    'ACO':       '#F0E442',   # yellow
    'HS-HHO':    '#56B4E9',   # sky blue
    'Local-Only':'#999999',   # neutral gray
    'Cloud-Only':'#000000',   # black
}
MARKERS: Dict[str, str] = {
    'DQN-ES':    'o',
    'PSO+DQN':   'h',         # hexagon — attribution control
    'ES-only':   's',
    'DQN-only':  'D',
    'PSO':       '^',
    'ACO':       'v',
    'HS-HHO':    'P',
    'Local-Only':'X',
    'Cloud-Only':'*',
}
# Default order across every figure
DEFAULT_ORDER: List[str] = [
    'DQN-ES', 'PSO+DQN', 'ES-only', 'DQN-only',
    'PSO', 'ACO', 'HS-HHO',
    'Local-Only', 'Cloud-Only',
]


def _style_for(alg: str) -> dict:
    """Per-algorithm style overrides.  DQN-ES and PSO+DQN are emphasised."""
    if alg == 'DQN-ES':
        return dict(linewidth=1.8, markersize=5.5, zorder=10,
                    markerfacecolor=COLORS[alg],
                    markeredgecolor=COLORS[alg])
    if alg == 'PSO+DQN':
        return dict(linewidth=1.4, markersize=5.0, zorder=9,
                    linestyle='--',
                    markerfacecolor=COLORS[alg],
                    markeredgecolor=COLORS[alg])
    return dict(linewidth=0.9, markersize=4.0, zorder=5,
                markerfacecolor='white',
                markeredgecolor=COLORS.get(alg, '#888888'))


def _bar_legend_handles(algs: Sequence[str]) -> List[Patch]:
    """Legend handles that match the horizontal bar styling."""
    handles: List[Patch] = []
    for alg in algs:
        edge = 'black' if alg == 'DQN-ES' else 'none'
        lw = 1.0 if alg == 'DQN-ES' else 0.0
        handles.append(Patch(facecolor=COLORS[alg], edgecolor=edge,
                             linewidth=lw, label=alg))
    return handles


def _scatter_legend_handles(algs: Sequence[str]) -> List[Line2D]:
    """Legend handles for Pareto scatter markers."""
    handles: List[Line2D] = []
    for alg in algs:
        is_bbodrl = (alg == 'DQN-ES')
        handles.append(Line2D(
            [0], [0],
            marker=MARKERS[alg],
            linestyle='',
            markerfacecolor=COLORS[alg],
            markeredgecolor='#222222' if is_bbodrl else COLORS[alg],
            markeredgewidth=1.2 if is_bbodrl else 0.9,
            markersize=7.0 if is_bbodrl else 6.0,
            label=alg,
        ))
    return handles


def _save(fig: plt.Figure, fdir: Path, name: str) -> None:
    fdir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(fdir / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[FIG] {name}.{{pdf,png}}')


def _load_summary(path: Path) -> Dict[int, dict]:
    with open(path, 'r', encoding='utf-8') as fh:
        raw = json.load(fh)
    return {int(k): v for k, v in raw.items()}


# ===========================================================================
# Figure 1 — latency vs task scale
# ===========================================================================
def fig_latency_vs_scale(summary, figures_dir, scales=None,
                          algs: Iterable[str] | None = None):
    """
    Mean latency vs. task scale on a log-x scale.  Shaded band = +/- 1 SD.
    Default trims to 6 series so the legend stays readable.
    """
    scales = scales or sorted(summary.keys())
    algs = algs or ['DQN-ES', 'ES-only', 'DQN-only',
                    'PSO', 'HS-HHO', 'Local-Only', 'Cloud-Only']

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for alg in algs:
        ys, es, xs = [], [], []
        for n in scales:
            d = summary.get(n, {}).get(alg, {}).get('avg_latency_ms')
            if d is None:
                continue
            ys.append(d['mean']); es.append(d['std']); xs.append(n)
        if not xs:
            continue
        x = np.array(xs); y = np.array(ys); e = np.array(es)
        s = _style_for(alg)
        ax.plot(x, y, color=COLORS[alg], marker=MARKERS[alg],
                label=alg, **s)
        ax.fill_between(x, y - e, y + e, color=COLORS[alg],
                        alpha=0.12, linewidth=0, zorder=s['zorder'] - 5)

    ax.set_xscale('log')
    ax.set_xticks(scales)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel('Number of tasks $N$')
    ax.set_ylabel('Average latency (ms)')
    ax.legend(loc='upper right', ncol=2, columnspacing=0.8,
              handletextpad=0.4, handlelength=1.4)
    _save(fig, figures_dir, 'fig1_latency_vs_scale')


# ===========================================================================
# Figure 2 — energy & SLA vs task scale (two-panel)
# ===========================================================================
def fig_energy_sla_vs_scale(summary, figures_dir, scales=None,
                             algs: Iterable[str] | None = None):
    scales = scales or sorted(summary.keys())
    algs = algs or ['DQN-ES', 'ES-only', 'DQN-only',
                    'PSO', 'HS-HHO', 'Local-Only', 'Cloud-Only']

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))
    for ax, key, ylabel in (
        (axes[0], 'avg_energy_mj',     'Energy per task (mJ)'),
        (axes[1], 'sla_violation_pct', 'SLA violations (%)'),
    ):
        for alg in algs:
            ys, es, xs = [], [], []
            for n in scales:
                d = summary.get(n, {}).get(alg, {}).get(key)
                if d is None:
                    continue
                ys.append(d['mean']); es.append(d['std']); xs.append(n)
            if not xs:
                continue
            x = np.array(xs); y = np.array(ys); e = np.array(es)
            s = _style_for(alg)
            ax.plot(x, y, color=COLORS[alg], marker=MARKERS[alg],
                    label=alg, **s)
            ax.fill_between(x, np.clip(y - e, 0, None), y + e,
                            color=COLORS[alg], alpha=0.12, linewidth=0,
                            zorder=s['zorder'] - 5)
        ax.set_xscale('log')
        ax.set_xticks(scales)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel('Number of tasks $N$')
        ax.set_ylabel(ylabel)
    # Single legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center',
               ncol=len(labels), bbox_to_anchor=(0.5, -0.03),
               columnspacing=0.9, handletextpad=0.4, handlelength=1.4)
    fig.subplots_adjust(wspace=0.28, bottom=0.22)
    _save(fig, figures_dir, 'fig2_energy_sla_vs_scale')


# ===========================================================================
# Helper: horizontal grouped bar panel
# ===========================================================================
def _hbar_panel(ax, algs, values, errors, xlabel, *,
                 show_yticklabels=True):
    """One horizontal-bar panel. Algorithm names on the y-axis avoid the
    rotated-x-label collisions that wreck the previous vertical-bar
    layout."""
    y = np.arange(len(algs))
    face = [COLORS[a] for a in algs]
    edge = ['black' if a == 'DQN-ES' else 'none' for a in algs]
    lw   = [1.0 if a == 'DQN-ES' else 0.0 for a in algs]
    ax.barh(y, values, xerr=errors, color=face, edgecolor=edge,
            linewidth=lw, height=0.62, capsize=2,
            error_kw={'linewidth': 0.6, 'ecolor': '#444444'})
    ax.set_yticks(y)
    if show_yticklabels:
        ax.set_yticklabels(algs, fontsize=7)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_xlim(left=0)


# ===========================================================================
# Figure 3 — metric comparison at primary scale (horizontal bars, 2x2)
# ===========================================================================
def fig_metric_bars(summary, figures_dir, ref_scale=PRIMARY_SCALE,
                     algs: Iterable[str] | None = None):
    """
    Four horizontal-bar panels in a 2x2 grid.  Algorithm names sit on the
    y-axis of the left column and are *hidden* on the right column so the
    two columns share a single labelled axis — that frees about 25 mm of
    horizontal space per panel and removes the rotated-tick collisions
    that plagued the previous version.
    """
    algs = list(algs or DEFAULT_ORDER)
    algs = [a for a in algs if a in summary.get(ref_scale, {})]

    metrics = [
        ('avg_latency_ms',    'Latency (ms)'),
        ('avg_energy_mj',     'Energy per task (mJ)'),
        ('avg_privacy_risk',  'Privacy risk'),
        ('sla_violation_pct', 'SLA violations (%)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8), sharey=True)
    for idx, (ax, (key, label)) in enumerate(zip(axes.flat, metrics)):
        vals = [summary[ref_scale][a][key]['mean'] for a in algs]
        errs = [summary[ref_scale][a][key]['std']  for a in algs]
        _hbar_panel(ax, algs, vals, errs, label,
                    show_yticklabels=(idx % 2 == 0))
    handles = _bar_legend_handles(algs)
    ncol = 5 if len(handles) >= 9 else 4 if len(handles) >= 7 else len(handles)
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=ncol,
               columnspacing=0.9, handletextpad=0.4,
               handlelength=1.4)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, figures_dir, 'fig3_metric_bars')


# ===========================================================================
# Figure 4 — Pareto frontier (energy vs latency)
# ===========================================================================
def _pareto_mask(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    n = len(xs)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if xs[j] <= xs[i] and ys[j] <= ys[i] and (
                    xs[j] < xs[i] or ys[j] < ys[i]):
                keep[i] = False
                break
    return keep


def _fig_pareto(summary, ref_scale, x_key, y_key, x_label, y_label,
                 out_name, figures_dir, algs):
    """
    Pareto scatter with a compact legend strip below the plot.  Markers
    are filled for visibility and use heavier glyphs for baseline nodes.
    """
    algs = [a for a in algs if a in summary.get(ref_scale, {})]
    xs, ys, names = [], [], []
    for alg in algs:
        d = summary[ref_scale][alg]
        x = d.get(x_key, {}).get('mean')
        y = d.get(y_key, {}).get('mean')
        if x is None or y is None:
            continue
        xs.append(x); ys.append(y); names.append(alg)
    xs_a = np.array(xs); ys_a = np.array(ys)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    for i, alg in enumerate(names):
        is_bbodrl = (alg == 'DQN-ES')
        ax.scatter(
            xs_a[i], ys_a[i],
            facecolors=COLORS[alg],
            edgecolors='#222222' if is_bbodrl else COLORS[alg],
            marker=MARKERS[alg],
            s=115 if is_bbodrl else 75,
            linewidths=1.2 if is_bbodrl else 0.9,
            zorder=9 if is_bbodrl else 7,
        )

    # Pareto front (minimisation of both axes)
    pareto_handle = None
    if len(xs_a) >= 3:
        mask = _pareto_mask(xs_a, ys_a)
        order = np.argsort(xs_a[mask])
        (pareto_handle,) = ax.plot(
            xs_a[mask][order], ys_a[mask][order],
            color='#666666', linewidth=0.8, linestyle='--', alpha=0.7,
            zorder=2, label='Pareto front'
        )

    # Generous padding so Local-Only / Cloud-Only markers do not sit on
    # the spine.
    x_pad = (xs_a.max() - xs_a.min()) * 0.10
    y_pad = (ys_a.max() - ys_a.min()) * 0.10
    ax.set_xlim(xs_a.min() - x_pad, xs_a.max() + x_pad)
    ax.set_ylim(ys_a.min() - y_pad, ys_a.max() + y_pad)

    ax.set_xlabel(x_label, labelpad=2)
    ax.set_ylabel(y_label, labelpad=2)
    handles = _scatter_legend_handles(names)
    if pareto_handle is not None:
        handles.append(pareto_handle)
    ncol = 5 if len(handles) >= 9 else 4 if len(handles) >= 7 else len(handles)
    ax.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.55), ncol=ncol,
              columnspacing=0.9, handletextpad=0.4,
              handlelength=1.2, frameon=False)
    fig.subplots_adjust(bottom=0.48)
    _save(fig, figures_dir, out_name)


def fig_pareto_energy_latency(summary, figures_dir, ref_scale=PRIMARY_SCALE):
    _fig_pareto(summary, ref_scale,
                x_key='avg_energy_mj', y_key='avg_latency_ms',
                x_label='Average energy per task (mJ)',
                y_label='Average latency (ms)',
                out_name='fig4_pareto_energy_latency',
                figures_dir=figures_dir,
                algs=DEFAULT_ORDER)


def fig_pareto_latency_privacy(summary, figures_dir, ref_scale=PRIMARY_SCALE):
    _fig_pareto(summary, ref_scale,
                x_key='avg_latency_ms', y_key='avg_privacy_risk',
                x_label='Average latency (ms)',
                y_label='Average privacy risk',
                out_name='fig5_pareto_latency_privacy',
                figures_dir=figures_dir,
                algs=DEFAULT_ORDER)


# ===========================================================================
# Figure 6 — epsilon convergence
# ===========================================================================
def fig_epsilon_convergence(trajectory_path: Path, figures_dir: Path):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    # Analytic envelope
    T = max(EPSILON_T_AT_0_05 * 2, 2000)
    t_anal = np.arange(1, T + 1)
    eps_anal = np.maximum(EPSILON_INIT * (EPSILON_DECAY ** t_anal),
                          EPSILON_MIN)
    ax.plot(t_anal, eps_anal, color='#444444', linestyle='--',
            linewidth=0.8, label='Analytic schedule')

    # Measured trajectories
    if trajectory_path and Path(trajectory_path).exists():
        try:
            with open(trajectory_path, 'r', encoding='utf-8') as fh:
                traj = json.load(fh)
            arrs = [np.array(v, dtype=float) for v in traj.values()
                    if isinstance(v, list) and len(v) > 0]
            if arrs:
                T_meas = min(len(a) for a in arrs)
                stack = np.array([a[:T_meas] for a in arrs])
                mu = stack.mean(axis=0); sd = stack.std(axis=0)
                t = np.arange(1, T_meas + 1)
                ax.plot(t, mu, color=COLORS['DQN-ES'], linewidth=1.6,
                        label=f'Measured (n={stack.shape[0]})')
                ax.fill_between(t, np.clip(mu - sd, 0, None), mu + sd,
                                color=COLORS['DQN-ES'], alpha=0.18,
                                linewidth=0)
        except Exception:
            pass

    ax.axhline(0.10, color='#888888', linewidth=0.4, linestyle=':')
    ax.axhline(0.05, color='#888888', linewidth=0.4, linestyle=':')
    ax.set_xlabel('Tasks processed')
    ax.set_ylabel(r'$\varepsilon$ (exploration probability)')
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc='upper right')
    _save(fig, figures_dir, 'fig6_epsilon_convergence')


# ===========================================================================
# Figure 7 — weight-scheme ablation (2x2 horizontal bars)
# ===========================================================================
def _draw_weight_ablation_panel(axes_row, data, modes, labels, palette,
                                metric_keys, panel_title=None):
    """Draw one 1×4 row of weight-ablation bars into axes_row."""
    for col_idx, (ax, (key, label)) in enumerate(zip(axes_row, metric_keys)):
        mu = [data[m][key]['mean'] for m in modes if m in data]
        sd = [data[m][key]['std']  for m in modes if m in data]
        present = [m for m in modes if m in data]
        y = np.arange(len(present))
        ax.barh(y, mu, xerr=sd, color=palette[:len(present)],
                edgecolor=['black' if m == 'nonlinear' else 'none'
                            for m in present],
                linewidth=[1.0 if m == 'nonlinear' else 0.0
                            for m in present],
                height=0.65, capsize=2,
                error_kw={'linewidth': 0.6, 'ecolor': '#444444'})
        ax.set_yticks(y)
        if col_idx == 0:
            ax.set_yticklabels([m.capitalize() for m in present], fontsize=7)
            if panel_title:
                ax.set_ylabel(panel_title, fontsize=7, labelpad=4)
        else:
            ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_xlim(left=0)


def fig_weight_ablation(
    mixed_path: Path,
    figures_dir: Path,
    highci_path: Path | None = None,
):
    """
    Two-panel weight-scheme ablation.

    Top row: mixed-CI workload (20 % high, 60 % medium, 20 % low).
    Bottom row: all-high-CI ICU scenario (Phi in [0.8, 1.0]).

    If highci_path is None or does not exist, only the mixed-CI panel is drawn.
    """
    if not mixed_path.exists():
        print(f'[FIG] Skipping weight ablation: {mixed_path} not found')
        return
    with open(mixed_path, 'r', encoding='utf-8') as fh:
        mixed_data = json.load(fh)

    highci_data = None
    if highci_path and highci_path.exists():
        with open(highci_path, 'r', encoding='utf-8') as fh:
            highci_data = json.load(fh)

    modes = ['flat', 'step', 'linear', 'nonlinear']
    metric_keys = [
        ('avg_latency_ms',    'Latency (ms)'),
        ('avg_energy_mj',     'Energy per task (mJ)'),
        ('avg_privacy_risk',  'Privacy risk'),
        ('sla_violation_pct', 'SLA violations (%)'),
    ]
    palette = ['#999999', '#56B4E9', '#009E73', COLORS['DQN-ES']]

    n_rows = 2 if highci_data is not None else 1
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(10.0, 3.0 * n_rows),
        sharey='row',
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # ensure 2D indexing

    _draw_weight_ablation_panel(
        axes[0], mixed_data, modes, None, palette, metric_keys,
        panel_title='Mixed-CI\n(20/60/20%)',
    )
    if highci_data is not None:
        _draw_weight_ablation_panel(
            axes[1], highci_data, modes, None, palette, metric_keys,
            panel_title='All-High-CI\n(Φ∈[0.8,1.0])',
        )

    scheme_labels = {
        'flat': 'Flat',
        'step': 'Step',
        'linear': 'Linear',
        'nonlinear': 'Non-linear',
    }
    scheme_handles = [
        Patch(
            facecolor=palette[i],
            edgecolor='black' if modes[i] == 'nonlinear' else 'none',
            linewidth=1.0 if modes[i] == 'nonlinear' else 0.0,
            label=scheme_labels[modes[i]],
        )
        for i in range(len(modes))
    ]
    fig.legend(handles=scheme_handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=4,
               columnspacing=0.9, handletextpad=0.4,
               handlelength=1.4)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    _save(fig, figures_dir, 'fig7_weight_ablation')


# ===========================================================================
# Figure 8 — Privacy Guard ROC
# ===========================================================================
def fig_privacy_guard_roc(metrics_path: Path, figures_dir: Path):
    """
    Single-curve ROC loaded directly from privacy_guard_metrics.json.

    Uses the roc_fpr / roc_tpr arrays saved by privacy_guard.py and the
    AUC value computed via sklearn.metrics.roc_auc_score (or trapezoidal
    fallback).  This eliminates the previous discrepancy where figures_q1.py
    recomputed AUC from a fixed Beta synthetic model while the text cited the
    value from the actual simulation.
    """
    if not metrics_path.exists():
        print(f'[FIG] Skipping ROC: {metrics_path} not found')
        return
    with open(metrics_path, 'r', encoding='utf-8') as fh:
        m = json.load(fh)

    # Load the actual ROC curve from the JSON instead of recomputing
    fpr_arr = np.array(m.get('roc_fpr', []), dtype=float)
    tpr_arr = np.array(m.get('roc_tpr', []), dtype=float)

    if fpr_arr.size == 0:
        # Fallback: reconstruct from stored operating-point scalars only
        fpr_arr = np.array([0.0, float(m.get('FPR', 0.05)), 1.0])
        tpr_arr = np.array([0.0, float(m.get('TPR (recall)', 0.98)), 1.0])

    auc = float(m.get('AUC', 0.0))

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    ax.plot(fpr_arr, tpr_arr, color=COLORS['DQN-ES'],
            linewidth=1.6, label=f'Privacy Guard (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#999999', linestyle='--',
            linewidth=0.6, label='Chance')
    # Operating point at the chosen threshold
    op_fpr = float(m.get('FPR', 0.0))
    op_tpr = float(m.get('TPR (recall)', 0.0))
    ax.scatter([op_fpr], [op_tpr], color='black', s=35, zorder=6,
               label=(f'$\\tau={m.get("threshold", 0.55):.2f}$: '
                      f'TPR={op_tpr:.3f}, FPR={op_fpr:.3f}'))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right')
    _save(fig, figures_dir, 'fig8_privacy_guard_roc')


# ===========================================================================
# Figure 9 — SHAP summary (bar form)
# ===========================================================================
def fig_shap_summary(shap_path: Path, figures_dir: Path):
    if not shap_path.exists():
        print(f'[FIG] Skipping SHAP: {shap_path} not found')
        return
    with open(shap_path, 'r', encoding='utf-8') as fh:
        d = json.load(fh)

    feats = d.get('feature_importances', {})
    if not feats:
        return
    # Sort by mean |SHAP|, descending
    items = sorted(feats.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals  = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    y = np.arange(len(names))
    ax.barh(y, vals, color=COLORS['DQN-ES'], height=0.6,
            edgecolor='none')
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(r'Mean $|\mathrm{SHAP}|$ on CI prediction')
    r2 = d.get('r2_score')
    if r2 is not None:
        ax.text(0.98, 0.05, f'$R^2={r2:.3f}$', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7,
                bbox=dict(facecolor='white', edgecolor='#bbbbbb',
                          boxstyle='round,pad=0.25', linewidth=0.5))
    _save(fig, figures_dir, 'fig9_shap_summary')


# ===========================================================================
# Figure 10 — MIT-BIH real-trace evaluation (2x2 horizontal bars)
# ===========================================================================
def fig_mitbih_trace(trace_path: Path, figures_dir: Path,
                      algs: Iterable[str] | None = None):
    """
    Same horizontal-bar layout as Fig.~3 so the two main comparison
    figures read identically.  Local-Only on the latency panel dwarfs
    the other bars (625 ms vs ~50 ms); we let it run off the right
    edge with a clipped tick label rather than rescaling everything to
    log, because the visual contrast is the point.
    """
    if not trace_path.exists():
        print(f'[FIG] Skipping MIT-BIH: {trace_path} not found')
        return
    with open(trace_path, 'r', encoding='utf-8') as fh:
        d = json.load(fh)

    algs = list(algs or DEFAULT_ORDER)
    algs = [a for a in algs if a in d]

    metrics = [
        ('avg_latency_ms',    'Latency (ms)'),
        ('avg_energy_mj',     'Energy per task (mJ)'),
        ('avg_privacy_risk',  'Privacy risk'),
        ('sla_violation_pct', 'SLA violations (%)'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8), sharey=True)
    for idx, (ax, (key, label)) in enumerate(zip(axes.flat, metrics)):
        vals = [d[a][key]['mean'] for a in algs]
        errs = [d[a][key]['std']  for a in algs]
        _hbar_panel(ax, algs, vals, errs, label,
                    show_yticklabels=(idx % 2 == 0))
    handles = _bar_legend_handles(algs)
    ncol = 5 if len(handles) >= 9 else 4 if len(handles) >= 7 else len(handles)
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=ncol,
               columnspacing=0.9, handletextpad=0.4,
               handlelength=1.4)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, figures_dir, 'fig10_mitbih_trace')

# ===========================================================================
# Driver
# ===========================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, default=None)
    p.add_argument('--figures-dir', type=str, default=None)
    p.add_argument('--ref-scale',   type=int, default=PRIMARY_SCALE)
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    res = Path(args.results_dir) if args.results_dir else project_root / 'results'
    fig = Path(args.figures_dir) if args.figures_dir else project_root / 'latex' / 'figures'

    summary_path = res / 'mc_full_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = _load_summary(summary_path)
    scales = sorted(summary.keys())

    print(f'[FIG] Writing figures into {fig}')
    fig_latency_vs_scale(summary, fig, scales=scales)
    fig_energy_sla_vs_scale(summary, fig, scales=scales)
    fig_metric_bars(summary, fig, ref_scale=args.ref_scale)
    fig_pareto_energy_latency(summary, fig, ref_scale=args.ref_scale)
    fig_pareto_latency_privacy(summary, fig, ref_scale=args.ref_scale)
    fig_epsilon_convergence(res / 'epsilon_trajectory.json', fig)
    # Two-panel weight ablation (mixed-CI + all-high-CI)
    fig_weight_ablation(
        res / 'weight_ablation_raw.json', fig,
        highci_path=res / 'weight_ablation_highci_raw.json',
    )
    fig_privacy_guard_roc(res / 'privacy_guard_metrics.json', fig)
    fig_shap_summary(res / 'shap_feature_importance.json', fig)
    fig_mitbih_trace(res / 'mitbih_trace_raw.json', fig)
    print('[FIG] Done.')


if __name__ == '__main__':
    main()
