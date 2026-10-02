"""
run_weight_ablation.py — Standalone weight-scheme ablation runner.

Usage:
    python run_weight_ablation.py               # mixed-CI workload (default)
    python run_weight_ablation.py --ci all_high # all-high-CI ICU scenario

Outputs go to results/:
    mixed:    table5_weight_ablation.csv, weight_ablation_raw.json
    all_high: table6_highci_weights.csv, weight_ablation_highci_raw.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PROJECT_ROOT = Path(BASE).parent
RESULTS_DIR  = PROJECT_ROOT / 'results'


def main():
    p = argparse.ArgumentParser(
        description='CI weight scheme ablation for DQN-ES.'
    )
    p.add_argument('--ci', choices=['mixed', 'all_high'], default='mixed',
                   help='"mixed" = standard workload; "all_high" = ICU scenario (Fix B)')
    p.add_argument('--n_runs',  type=int, default=30)
    p.add_argument('--n_tasks', type=int, default=1000)
    p.add_argument('--workers', type=int, default=None)
    args = p.parse_args()

    from src.analysis.weight_ablation import run_ablation
    print(f'[run_weight_ablation] ci_distribution={args.ci!r}  '
          f'n_tasks={args.n_tasks}  n_runs={args.n_runs}')
    run_ablation(args.n_tasks, args.n_runs, RESULTS_DIR,
                 ci_distribution=args.ci, workers=args.workers)


if __name__ == '__main__':
    main()
