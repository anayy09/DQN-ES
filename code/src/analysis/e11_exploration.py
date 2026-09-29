"""
e11_exploration.py — exploration and off-policy diagnostics for the DQN
schedulers, from the per-task raw logs (plan E11).

Per scheduler with DQN diagnostics (rows where exec_q_rank is logged):
  mismatch          fraction of decisions where the executed destination x*
                    differs from argmax-Q, overall, on exploratory and on
                    greedy decisions
  by epsilon phase  epsilon > 0.5, 0.1 < epsilon <= 0.5, epsilon <= 0.1:
                    decisions, exploratory share, mismatch share, and the
                    executed-destination distribution
  Q-rank histogram  rank of x* in the Q ordering (0 = argmax)
  TD-loss / Q curves mean td_loss and q_max per block of 50 decisions,
                    averaged over replicates (td_loss is the loss of the
                    minibatch update performed at that decision)

Outputs (<out>/): e11_summary.csv, e11_by_phase.csv, e11_qrank.csv,
e11_curves.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

PHASES = [('eps>0.5', 0.5, 1.01), ('0.1<eps<=0.5', 0.1, 0.5), ('eps<=0.1', -1.0, 0.1)]
BLOCK = 50


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def _write(path, rows):
    from src.simulation.episode_log import _long_path
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(_long_path(path), 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, restval='')
        wr.writeheader()
        wr.writerows(rows)


def run(raw_dirs, out_dir: Path, algorithms=None) -> dict:
    from src.analysis.e5_adversary import _dest, load_logs
    from src.simulation.episode_log import _long_path
    logs = load_logs([Path(d) for d in raw_dirs], algorithms)
    summary, phases, qrank, curves = [], [], [], []
    for alg, runs in sorted(logs.items()):
        rows = [r for rs in runs.values() for r in rs if r.get('exec_q_rank') not in ('', None)]
        if not rows:
            continue
        expl = np.array([r['explored'] == '1' for r in rows])
        mism = np.array([r['assigned_node'] != r['q_argmax_node'] for r in rows])
        summary.append({'scheduler': alg, 'n_decisions': len(rows), 'n_runs': len(runs),
                        'explored_frac': float(expl.mean()),
                        'mismatch_frac': float(mism.mean()),
                        'mismatch_frac_greedy': float(mism[~expl].mean()) if (~expl).any() else float('nan'),
                        'mismatch_frac_explore': float(mism[expl].mean()) if expl.any() else float('nan')})
        eps = np.array([_f(r['epsilon']) for r in rows])
        for name, lo, hi in PHASES:
            sel = (eps > lo) & (eps <= hi)
            if not sel.any():
                continue
            dist = Counter(_dest(r) for r, s in zip(rows, sel) if s)
            tot = sum(dist.values())
            phases.append({'scheduler': alg, 'phase': name, 'n_decisions': int(sel.sum()),
                           'explored_frac': float(expl[sel].mean()),
                           'mismatch_frac': float(mism[sel].mean()),
                           **{f'dest_{d}': c / tot for d, c in sorted(dist.items())}})
        ranks = Counter(int(r['exec_q_rank']) for r in rows)
        for k in sorted(ranks):
            qrank.append({'scheduler': alg, 'q_rank': k, 'count': ranks[k],
                          'frac': ranks[k] / len(rows)})
        # curves per block of decisions, averaged over replicates
        per_block = defaultdict(lambda: {'td_loss': [], 'q_max': []})
        for rs in runs.values():
            for i, r in enumerate(rs):
                b = i // BLOCK
                for k in ('td_loss', 'q_max'):
                    v = _f(r.get(k))
                    if not np.isnan(v):
                        per_block[b][k].append(v)
        for b in sorted(per_block):
            d = per_block[b]
            curves.append({'scheduler': alg, 'decision_block_start': b * BLOCK,
                           'td_loss_mean': float(np.mean(d['td_loss'])) if d['td_loss'] else '',
                           'q_max_mean': float(np.mean(d['q_max'])) if d['q_max'] else ''})
    out_dir = Path(out_dir)
    _long_path(out_dir).mkdir(parents=True, exist_ok=True)
    _write(out_dir / 'e11_summary.csv', summary)
    _write(out_dir / 'e11_by_phase.csv', phases)
    _write(out_dir / 'e11_qrank.csv', qrank)
    _write(out_dir / 'e11_curves.csv', curves)
    for s in summary:
        print(f"  {s['scheduler']:<16} explored {s['explored_frac']:.3f}  x*!=argmaxQ "
              f"{s['mismatch_frac']:.3f} (greedy {s['mismatch_frac_greedy']:.3f})")
    return {'summary': summary, 'phases': phases, 'qrank': qrank, 'curves': curves}


def main():
    p = argparse.ArgumentParser(description='E11 exploration diagnostics')
    p.add_argument('--raw-dirs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--algorithms', nargs='+', default=None)
    a = p.parse_args()
    run(a.raw_dirs, Path(a.out), a.algorithms)


if __name__ == '__main__':
    main()
