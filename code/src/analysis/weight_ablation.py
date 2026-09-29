"""
weight_ablation.py — Fix 6 + Fix B: CI weight function ablation.

Compares four CI-to-weight schemes (flat, step, linear, proposed
non-linear) for DQN-ES at the primary scale, 30 Monte Carlo trials each.

Fix B: also runs all-high-CI ICU scenario (Phi in [0.8, 1.0]) to test
whether the non-linear scheme separates from alternatives under maximum
criticality load.  Wilcoxon rank-sum tests (Bonferroni-corrected) between
non-linear and each other scheme are applied in both scenarios.

Outputs (mixed-CI workload):
  results/table5_weight_ablation.csv
  results/weight_ablation_raw.json

Outputs (all-high-CI workload, Fix B):
  results/table6_highci_weights.csv
  results/weight_ablation_highci_raw.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.algorithms.dqn_es import DQNESScheduler
from src.config import (
    GLOBAL_SEED,
    N_FOG_NODES,
    N_RUNS,
    N_WEARABLES,
    PRIMARY_SCALE,
)
from src.core.cost_function import get_weight_mode, set_weight_mode
from src.core.task import HealthcareTask
from src.simulation.replicate import build_synthetic_replicate, run_scheduler

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


WEIGHT_MODES = ['flat', 'step', 'linear', 'nonlinear']


def _run_once(payload):
    n_tasks, run_id, ci_distribution, mode = payload
    set_weight_mode(mode)
    try:
        seeds, topo, tasks = build_synthetic_replicate(run_id, n_tasks,
                                                       ci_distribution)
        res, _ = run_scheduler(DQNESScheduler, topo, tasks, seeds)
    finally:
        set_weight_mode('nonlinear')
    if not res:
        return None
    return {
        'run_id':            run_id,
        'avg_latency_ms':    mean(r['latency_ms']    for r in res),
        'avg_energy_mj':     mean(r['energy_mj']     for r in res),
        'avg_privacy_risk':  mean(r['privacy_risk']  for r in res),
        'sla_violation_pct': 100.0 * sum(r['sla_violated']
                                         for r in res) / len(res),
    }


def run_ablation(
    n_tasks: int,
    n_runs: int,
    out_dir: Path,
    ci_distribution: str = 'mixed',
    workers: int | None = None,
) -> dict:
    """
    Run the weight-scheme ablation.

    Parameters
    ----------
    ci_distribution : str
        'mixed' for standard workload; 'all_high' for Fix B ICU scenario.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    label = 'HIGHCI' if ci_distribution == 'all_high' else 'MIXED'
    print(f'[WEIGHT-AB] ci_distribution={ci_distribution} ({label})')

    raw: dict = defaultdict(list)
    import concurrent.futures
    payloads = [(n_tasks, run_id, ci_distribution, mode) for mode in WEIGHT_MODES for run_id in range(n_runs)]
    
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    
    print(f'[WEIGHT-AB] Dispatching {len(payloads)} tasks over {workers} workers...')
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        if _TQDM:
            results = list(tqdm(executor.map(_run_once, payloads), total=len(payloads), desc='  Ablation', ncols=70))
        else:
            results = list(executor.map(_run_once, payloads))
            
    for payload, m in zip(payloads, results):
        if m is not None:
            raw[payload[3]].append(m)
    set_weight_mode('nonlinear')

    # Aggregate
    summary = {}
    csv_rows = []
    for mode in WEIGHT_MODES:
        runs = raw[mode]
        row = {'weight_mode': mode}
        agg = {}
        for k in ['avg_latency_ms', 'avg_energy_mj',
                  'avg_privacy_risk', 'sla_violation_pct']:
            vs = np.array([r[k] for r in runs], dtype=float)
            agg[k] = {'mean': float(vs.mean()), 'std': float(vs.std()),
                      'samples': vs.tolist(),
                      'run_ids': [int(r['run_id']) for r in runs]}
            row[f'{k}_mean'] = float(vs.mean())
            row[f'{k}_std']  = float(vs.std())
        summary[mode] = agg
        csv_rows.append(row)

    # Paired signed-rank tests, non-linear vs each other scheme (declared
    # family 'weight_ablation' in src.config.STAT_FAMILIES; Holm over 12).
    # Imported here so spawned workers do not import scipy via eval_stats
    from src.analysis.statistical_tests import run_family_tests
    tests = run_family_tests(summary, 'weight_ablation')
    print('\n[WEIGHT-AB] Paired signed-rank tests (d = nonlinear - scheme), '
          f'Holm over {len(tests)} tests:')
    for t in tests:
        if t['status'] != 'tested':
            print(f"  {t['metric']} vs {t['comparator']}: {t['status']}")
            continue
        sig = '*' if t['reject_holm'] else ' '
        print(f"  {t['metric']} vs {t['comparator']}: d={t['hl_diff']:+.4f} "
              f"[{t['ci_lo']:+.4f}, {t['ci_hi']:+.4f}] p_holm={t['p_holm']:.4e}{sig}")
    for row in csv_rows:
        for t in tests:
            if t['comparator'] == row['weight_mode'] and t['status'] == 'tested':
                m = t['metric']
                row[f'{m}_hl_diff_nonlinear_minus'] = t['hl_diff']
                row[f'{m}_ci_lo'] = t['ci_lo']
                row[f'{m}_ci_hi'] = t['ci_hi']
                row[f'{m}_p_holm_vs_nonlinear'] = t['p_holm']

    # Determine output file names based on ci_distribution
    if ci_distribution == 'all_high':
        csv_name = 'table6_highci_weights.csv'
        json_name = 'weight_ablation_highci_raw.json'
    else:
        csv_name = 'table5_weight_ablation.csv'
        json_name = 'weight_ablation_raw.json'

    if csv_rows:
        with open(out_dir / csv_name, 'w', newline='', encoding='utf-8') as fh:
            fields = list(dict.fromkeys(k for r in csv_rows for k in r))
            wr = csv.DictWriter(fh, fieldnames=fields, restval='')
            wr.writeheader()
            wr.writerows(csv_rows)
    with open(out_dir / json_name, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    print('\n[WEIGHT-AB] Summary:')
    hdr = (f"  {'Mode':<10s} {'Lat(ms)':>10s} {'Eng(mJ)':>10s} "
           f"{'Priv':>8s} {'SLA%':>8s}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for mode in WEIGHT_MODES:
        s = summary[mode]
        print(f'  {mode:<10s} '
              f'{s["avg_latency_ms"]["mean"]:>10.2f} '
              f'{s["avg_energy_mj"]["mean"]:>10.4f} '
              f'{s["avg_privacy_risk"]["mean"]:>8.4f} '
              f'{s["sla_violation_pct"]["mean"]:>8.2f}')
    print(f'\n[WEIGHT-AB] Saved {csv_name} and {json_name} -> {out_dir}')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_runs', type=int, default=N_RUNS)
    p.add_argument('--n_tasks', type=int, default=PRIMARY_SCALE)
    p.add_argument('--output', type=str, default=None)
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    out_dir = (Path(args.output) if args.output
               else project_root / 'results')
    run_ablation(args.n_tasks, args.n_runs, out_dir)


if __name__ == '__main__':
    main()
