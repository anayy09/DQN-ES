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
from scipy import stats as scipy_stats

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
        'avg_latency_ms':    mean(r['latency_ms']    for r in res),
        'avg_energy_mj':     mean(r['energy_mj']     for r in res),
        'avg_privacy_risk':  mean(r['privacy_risk']  for r in res),
        'sla_violation_pct': 100.0 * sum(r['sla_violated']
                                         for r in res) / len(res),
    }


def _wilcoxon_vs_nonlinear(summary: dict) -> dict:
    """
    Fix B: Wilcoxon rank-sum tests between non-linear and each other scheme.
    Bonferroni correction: 3 schemes × 4 metrics = 12 tests.
    Returns dict[metric][scheme] = {p_raw, p_corrected}.
    """
    comparators = ['flat', 'step', 'linear']
    metric_keys = ['avg_latency_ms', 'avg_energy_mj',
                   'avg_privacy_risk', 'sla_violation_pct']
    n_tests = len(comparators) * len(metric_keys)   # 12
    alpha = 0.05

    if 'nonlinear' not in summary:
        return {}
    nl_samples = {k: np.array(summary['nonlinear'][k]['samples'], dtype=float)
                  for k in metric_keys}

    results = {}
    for metric in metric_keys:
        results[metric] = {}
        for scheme in comparators:
            if scheme not in summary:
                continue
            s2 = np.array(summary[scheme][metric]['samples'], dtype=float)
            try:
                _, p_raw = scipy_stats.ranksums(nl_samples[metric], s2)
            except Exception:
                p_raw = float('nan')
            p_corr = min(float(p_raw) * n_tests, 1.0) if not np.isnan(p_raw) else float('nan')
            results[metric][scheme] = {'p_raw': float(p_raw), 'p_corrected': float(p_corr)}
    return results


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
                      'samples': vs.tolist()}
            row[f'{k}_mean'] = float(vs.mean())
            row[f'{k}_std']  = float(vs.std())
        summary[mode] = agg
        csv_rows.append(row)

    # Wilcoxon tests between non-linear and other schemes
    wilcoxon_results = _wilcoxon_vs_nonlinear(summary)
    if wilcoxon_results:
        print('\n[WEIGHT-AB] Wilcoxon tests (non-linear vs. others), '
              'Bonferroni-corrected (12 tests):')
        for metric, comps in wilcoxon_results.items():
            for scheme, r in comps.items():
                sig = '*' if r['p_corrected'] < 0.05 else ' '
                print(f'  {metric} vs {scheme}: p_corr={r["p_corrected"]:.4e}{sig}')
        # Append Wilcoxon results to csv_rows
        for mode in WEIGHT_MODES:
            for row in csv_rows:
                if row['weight_mode'] == mode:
                    for metric, comps in wilcoxon_results.items():
                        if mode in comps:
                            row[f'{metric}_p_corr_vs_nonlinear'] = \
                                comps[mode].get('p_corrected', float('nan'))

    # Determine output file names based on ci_distribution
    if ci_distribution == 'all_high':
        csv_name = 'table6_highci_weights.csv'
        json_name = 'weight_ablation_highci_raw.json'
    else:
        csv_name = 'table5_weight_ablation.csv'
        json_name = 'weight_ablation_raw.json'

    if csv_rows:
        with open(out_dir / csv_name, 'w', newline='', encoding='utf-8') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
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
