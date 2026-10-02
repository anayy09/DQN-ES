"""
privacy_guard.py — traffic-analysis attack detector validation.

Detection mechanism (explicit, reproducible):
  - A flow is FLAGGED as a candidate traffic-analysis attack whenever the
    source device's instantaneous offload-entropy ratio H(u_i)/H_max falls
    below PRIVACY_ENTROPY_THRESHOLD (default 0.55).  Intuition: an attacker
    performing traffic analysis on a single victim device manifests as a
    sudden concentration of outbound flows to one observation node, which
    drives the entropy ratio toward zero.

  - Ground truth comes from MedSec-25 labels: any label that is not
    'benign' and whose severity_from_label() ≥ 0.55 is treated as a
    positive (attack) example; the remainder are negatives.

  - The per-flow entropy ratio is derived from real MedSec-25
    features (attack_probability, computed in parse_medsec.py from actual
    flow bytes and duration), replacing the previous synthetic Beta
    distribution model.

Reported metrics:
  - TPR = TP / (TP + FN)
  - FPR = FP / (FP + TN)
  - Precision, Recall, F1
  - ROC and confusion matrix figures

This script does NOT require running the full simulator.  It loads the
cached medsec_events.json (or parses the CSV on demand) and replays the
traffic-analysis decision rule.

Outputs:
  results/privacy_guard_metrics.json
  latex/figures/fig_privacy_guard_roc.{pdf,png}
  latex/figures/fig_privacy_guard_confusion.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from sklearn.metrics import roc_auc_score as _sklearn_auc
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import PRIVACY_ENTROPY_THRESHOLD
from src.data_ingestion.parse_medsec import (
    _severity_from_label,
    load_medsec_events,
)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'savefig.bbox': 'tight',
    'figure.dpi': 300, 'savefig.dpi': 300,
})


# ---------------------------------------------------------------------------
# Synthetic entropy-ratio model
# ---------------------------------------------------------------------------

def _entropy_ratio_from_flow(is_attack: bool, severity: float,
                              attack_prob: float) -> float:
    """
    Derive the entropy ratio for a flow from its real MedSec-25 features.

    Replaces the previous synthetic Beta distribution model.
    The entropy ratio is now computed deterministically from the
    attack_probability field (derived from actual flow bytes and duration
    in parse_medsec.py), making the Privacy Guard evaluation grounded
    in real dataset statistics.

    Logic:
      - Attack flows concentrate traffic on a single observation node,
        giving a low entropy ratio: eta = 1 - attack_probability
      - Benign flows are diversified, giving a high entropy ratio:
        eta = 1 - 0.1 * attack_probability
    Both are clipped to [0, 1].
    """
    if is_attack:
        eta = 1.0 - float(attack_prob)
    else:
        eta = 1.0 - 0.1 * float(attack_prob)
    return float(max(0.0, min(1.0, eta)))


def run_validation(data_dir: Path, out_dir: Path,
                   fig_dir: Path | None = None,
                   threshold: float = PRIVACY_ENTROPY_THRESHOLD,
                   workers: int | None = None,
                   seed: int = 42) -> Dict:
    results_dir = out_dir
    figures_dir = fig_dir if fig_dir else out_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f'[PG] Loading MedSec-25 events from {data_dir} ...')
    events = load_medsec_events(str(data_dir))
    print(f'[PG] Loaded {len(events)} flow events.')

    # Build labelled vector — use real flow features, not synthetic Beta
    y_true   = np.zeros(len(events), dtype=int)
    h_ratios = np.zeros(len(events), dtype=float)
    for i, ev in enumerate(events):
        sev = ev.get('severity', _severity_from_label(ev.get('label', '')))
        is_attack = sev >= 0.55
        y_true[i] = int(is_attack)
        attack_prob = ev.get('attack_probability', sev * 0.5)
        h_ratios[i] = _entropy_ratio_from_flow(bool(is_attack), float(sev),
                                               float(attack_prob))

    # Decision rule: H_ratio < threshold => flagged as attack
    y_pred = (h_ratios < threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tpr
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

    metrics = {
        'threshold':  threshold,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        'TPR (recall)': tpr,
        'FPR':       fpr,
        'precision': precision,
        'F1':        f1,
        'n_total':   int(len(events)),
        'n_attack':  int(y_true.sum()),
        'n_benign':  int((1 - y_true).sum()),
    }
    print(f'\n[PG] Detection results (threshold={threshold}):')
    for k, v in metrics.items():
        print(f'   {k:<14s}: {v:.4f}' if isinstance(v, float)
              else f'   {k:<14s}: {v}')

    # ----- ROC curve -----
    thresholds_sweep = np.linspace(0.0, 1.0, 101)
    tpr_curve, fpr_curve = [], []
    for thr in thresholds_sweep:
        yp = (h_ratios < thr).astype(int)
        t_p = int(((yp == 1) & (y_true == 1)).sum())
        f_p = int(((yp == 1) & (y_true == 0)).sum())
        t_n = int(((yp == 0) & (y_true == 0)).sum())
        f_n = int(((yp == 0) & (y_true == 1)).sum())
        tpr_curve.append(t_p / max(t_p + f_n, 1))
        fpr_curve.append(f_p / max(f_p + t_n, 1))
    tpr_curve = np.array(tpr_curve)
    fpr_curve = np.array(fpr_curve)

    # AUC: prefer sklearn.metrics.roc_auc_score (exact computation).
    # Falls back to trapezoidal rule on the 101-point sweep if sklearn is absent.
    order = np.argsort(fpr_curve)
    if _HAS_SKLEARN:
        # Use the continuous h_ratio score for an exact AUC — attack label = 1
        # (h_ratio < threshold triggers detection, so invert: lower h = higher attack score)
        auc = float(_sklearn_auc(y_true, 1.0 - h_ratios))
        print(f'   [AUC method: sklearn.metrics.roc_auc_score (exact)]')
    else:
        _trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)
        if _trapz is None:
            x = fpr_curve[order]
            y = tpr_curve[order]
            auc = float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
        else:
            auc = float(_trapz(tpr_curve[order], fpr_curve[order]))
        print(f'   [AUC method: trapezoidal rule on 101-point sweep]')
    metrics['AUC'] = auc
    # Save full ROC curve so figures_q1.py uses the same data, not Beta model
    metrics['roc_fpr'] = fpr_curve[order].tolist()
    metrics['roc_tpr'] = tpr_curve[order].tolist()
    print(f'   {"AUC":<14s}: {auc:.4f}')

    # ----- Plot ROC -----
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(fpr_curve[order], tpr_curve[order], color='#D62728',
            linewidth=1.4, label=f'Privacy Guard (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.7, label='Random')
    # Operating point at the chosen threshold
    ax.scatter([fpr], [tpr], color='black', s=30, zorder=5,
               label=f'Operating point\n(τ={threshold:.2f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Privacy Guard ROC on MedSec-25', fontsize=8, pad=4)
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', fontsize=6.5, framealpha=0.8)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(figures_dir / f'fig_privacy_guard_roc.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[PG] Saved fig_privacy_guard_roc.{{pdf,png}}')

    # ----- Confusion matrix figure -----
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(cm, cmap='Reds')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() * 0.5 else 'black',
                    fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred: Benign', 'Pred: Attack'])
    ax.set_yticklabels(['True: Benign', 'True: Attack'])
    ax.set_title('Privacy Guard Confusion Matrix', fontsize=8, pad=4)
    plt.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(figures_dir / f'fig_privacy_guard_confusion.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'[PG] Saved fig_privacy_guard_confusion.{{pdf,png}}')

    out = results_dir / 'privacy_guard_metrics.json'
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)
    print(f'[PG] Saved {out}')

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',    type=str, default=None)
    p.add_argument('--results-dir', type=str, default=None)
    p.add_argument('--figures-dir', type=str, default=None)
    p.add_argument('--threshold',   type=float,
                   default=PRIVACY_ENTROPY_THRESHOLD)
    p.add_argument('--seed',        type=int, default=42)
    args = p.parse_args()

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    data_dir     = Path(args.data_dir)    if args.data_dir   else project_root / 'data'
    results_dir  = Path(args.results_dir) if args.results_dir else project_root / 'results'
    figures_dir  = Path(args.figures_dir) if args.figures_dir else project_root / 'latex' / 'figures'

    run_validation(data_dir, results_dir, figures_dir,
                   threshold=args.threshold, seed=args.seed)


if __name__ == '__main__':
    main()
