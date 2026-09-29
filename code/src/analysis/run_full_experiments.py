"""
run_full_experiments.py — Q1 master Monte Carlo driver.

Addresses Fix 1 (N_RUNS=30 globally enforced), Fix 2 (ES-only + DQN-only
ablations included), Fix 3 (Local-Only SLA violations now non-zero due to
realistic cycle counts in TASK_PROFILES), Fix 5 (N=5000 scale added), and
Fix 7 (epsilon trajectory captured for DQN-ES).

Outputs:
  results/mc_full_results.json          Per-run raw metrics
  results/mc_full_summary.json          Mean ± std per (scale, algorithm)
  results/epsilon_trajectory.json       DQN-ES epsilon vs. tasks processed
  results/table3_n1000.csv              Table III (primary comparison)

The orchestration script `run_q1_pipeline.py` then calls statistical_tests
and figure generators on the JSON outputs.

Usage
-----
    python -m src.analysis.run_full_experiments
    python -m src.analysis.run_full_experiments --n_runs 30
    python -m src.analysis.run_full_experiments --scales 100 1000 5000
    python -m src.analysis.run_full_experiments --quick  # 5 runs (debug only)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.config import (
    GLOBAL_SEED,
    N_FOG_NODES,
    N_RUNS,
    N_WEARABLES,
    PRIMARY_SCALE,
    TASK_SCALES,
    get_full_algorithm_registry,
    get_registry,
)
from src.core.task import HealthcareTask
from src.simulation.episode_log import (
    add_queue_metrics,
    add_steady_state,
    raw_log_path,
    write_raw_log,
)
from src.simulation.replicate import (
    build_synthetic_replicate,
    overrides_tag,
    run_scheduler,
)

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_single(alg_name, sched_cls, n_tasks, run_id, raw_dir=None,
                task_overrides=None, n_fog_nodes=N_FOG_NODES):
    """One Monte Carlo replicate for one algorithm (topology per replicate)."""
    seeds, topo, tasks = build_synthetic_replicate(
        run_id, n_tasks, 'mixed', n_fog_nodes=n_fog_nodes,
        task_overrides=task_overrides)
    results, sched = run_scheduler(
        sched_cls, topo, tasks, seeds,
        warm_start_tasks=(task_overrides or {}).get('warm_start_tasks'))
    if raw_dir is not None and results:
        write_raw_log(results, raw_log_path(raw_dir, n_tasks, alg_name, run_id),
                      alg_name, run_id, n_tasks)

    if not results:
        return {
            'avg_latency_ms': 0.0, 'avg_energy_mj': 0.0,
            'avg_privacy_risk': 0.0, 'sla_violation_pct': 0.0,
            'throughput': 0.0,
        }, None

    timestamps = [r['timestamp'] for r in results]
    span = max(timestamps) - min(timestamps) + 1e-6

    metrics = {
        'avg_latency_ms':    mean(r['latency_ms']    for r in results),
        'avg_energy_mj':     mean(r['energy_mj']     for r in results),
        'avg_privacy_risk':  mean(r['privacy_risk']  for r in results),
        'sla_violation_pct': 100.0 * sum(r['sla_violated'] for r in results)
                              / len(results),
        'throughput':        n_tasks / span,
    }
    add_steady_state(metrics, results)
    add_queue_metrics(metrics, results)
    metrics['warm_started'] = float(getattr(sched, 'warm_started_with', 0) > 0)

    epsilon_history = getattr(sched, 'epsilon_history', None)
    return metrics, (list(epsilon_history) if epsilon_history else None)

def _run_single_wrapper(args):
    alg, sched_cls, n_tasks, run_id, raw_dir, overrides, n_fog = args
    try:
        m, eps_hist = _run_single(alg, sched_cls, n_tasks, run_id, raw_dir,
                                  overrides, n_fog)
        m['run_id'] = run_id
        return run_id, m, eps_hist, None
    except Exception as exc:
        return run_id, None, None, str(exc)

# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def run_full(
    task_scales: list[int],
    n_runs: int,
    results_dir: Path,
    workers: int = None,
    registry_name: str = 'main',
    algorithms: list[str] | None = None,
    raw_logs: bool = True,
    task_overrides: dict | None = None,
    n_fog_nodes: int = N_FOG_NODES,
) -> Dict:
    """
    registry_name 'main' writes mc_full_*.json and table3_n{N}.csv;
    'experiments' / 'all' write mc_exp_* / mc_all_* and table_exp_n{N}.csv
    so the main-comparison files are never overwritten by decomposition arms.
    `algorithms` restricts the run to a subset of the registry.
    raw_logs writes one gzip CSV per run under results_dir/raw/<prefix>/.
    task_overrides: sensitivity knobs (ecg_payload_bits, result_size_bits),
    see simulation/replicate.apply_task_overrides.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    registry = get_registry(registry_name)
    if algorithms:
        missing = [a for a in algorithms if a not in registry]
        if missing:
            raise KeyError(f'not in registry {registry_name!r}: {missing}')
        registry = {a: registry[a] for a in algorithms}
    alg_names = list(registry.keys())
    prefix = {'main': 'mc_full', 'experiments': 'mc_exp',
              'all': 'mc_all'}[registry_name]
    table_name = ('table3' if registry_name == 'main'
                  else f'table_{registry_name[:3]}')
    metric_keys = ['avg_latency_ms', 'avg_energy_mj',
                   'avg_privacy_risk', 'sla_violation_pct', 'throughput',
                   'avg_privacy_risk_ss', 'avg_queue_ms', 'edge_utilisation']
    raw_dir = (results_dir / 'raw' / prefix) if raw_logs else None

    mc_raw:     Dict[int, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    mc_summary: Dict[int, Dict[str, dict]] = defaultdict(dict)
    eps_trajectories: Dict[int, List[float]] = {}

    print('=' * 72)
    print(f'[Q1-MC] Scales={task_scales} N_RUNS={n_runs} Algorithms={len(alg_names)}')
    print('=' * 72)

    t0 = time.time()
    import concurrent.futures
    for n_tasks in task_scales:
        print(f'\n[Q1-MC] Scale n={n_tasks}')
        
        # Batch all algorithms together to maximize CPU utilization
        args_list = []
        for alg in alg_names:
            sched_cls = registry[alg]
            for run_id in range(n_runs):
                args_list.append((alg, sched_cls, n_tasks, run_id, raw_dir,
                                  task_overrides, n_fog_nodes))
                
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            if _TQDM:
                it = tqdm(executor.map(_run_single_wrapper, args_list), total=len(args_list), desc=f'  Tasks {n_tasks}', leave=False, ncols=70)
            else:
                it = executor.map(_run_single_wrapper, args_list)
            
            for i, (run_id, m, eps_hist, err) in enumerate(it):
                alg = args_list[i][0]
                if err:
                    warnings.warn(f'[Q1-MC] {alg} n={n_tasks} run={run_id} failed: {err}')
                    continue
                mc_raw[n_tasks][alg].append(m)
                if (alg == 'DQN-ES' and n_tasks == PRIMARY_SCALE and eps_hist is not None):
                    eps_trajectories.setdefault(run_id, eps_hist)

        # Aggregate for all algorithms at this scale
        for alg in alg_names:
            agg = {}
            for key in metric_keys:
                runs = [r for r in mc_raw[n_tasks][alg] if key in r]
                vals = np.array([r[key] for r in runs], dtype=float)
                if len(vals):
                    agg[key] = {
                        'mean': float(vals.mean()),
                        'std':  float(vals.std()),
                        'min':  float(vals.min()),
                        'max':  float(vals.max()),
                        'n':    int(len(vals)),
                        'samples': [float(v) for v in vals],
                        'run_ids': [int(r['run_id']) for r in runs],
                    }
                elif key == 'avg_privacy_risk_ss':
                    continue   # undefined: no device passed the W-decision warm-up
                else:
                    agg[key] = {'mean': 0.0, 'std': 0.0, 'min': 0.0,
                                'max': 0.0, 'n': 0, 'samples': [],
                                'run_ids': []}
            mc_summary[n_tasks][alg] = agg

        _print_table(n_tasks, mc_summary[n_tasks], alg_names)

    elapsed = time.time() - t0
    print(f'\n[Q1-MC] Total wall time: {elapsed:.1f}s')

    # Persist
    _save_json(mc_raw,     results_dir / f'{prefix}_results.json')
    _save_json(mc_summary, results_dir / f'{prefix}_summary.json')

    if eps_trajectories and registry_name == 'main':
        _save_json(eps_trajectories,
                   results_dir / 'epsilon_trajectory.json')
        print(f'[Q1-MC] Captured epsilon trajectories for '
              f'{len(eps_trajectories)} runs at n={PRIMARY_SCALE}.')

    # Table III CSV at primary scale
    _save_table3_csv(mc_summary, alg_names,
                     results_dir / f'{table_name}_n{PRIMARY_SCALE}.csv')
    return mc_summary


def _print_table(n_tasks, scale, alg_names):
    hdr = (f"{'Algorithm':<18} {'Lat ms':>10} {'Eng mJ':>10} "
           f"{'Priv':>8} {'SLA%':>8} {'Tput':>10}")
    print(f'  {hdr}\n  {"-"*len(hdr)}')
    for alg in alg_names:
        d = scale.get(alg, {})
        lat = d.get('avg_latency_ms', {}).get('mean', 0)
        eng = d.get('avg_energy_mj', {}).get('mean', 0)
        prv = d.get('avg_privacy_risk', {}).get('mean', 0)
        sla = d.get('sla_violation_pct', {}).get('mean', 0)
        thr = d.get('throughput', {}).get('mean', 0)
        tag = '*' if alg == 'DQN-ES' else ' '
        print(f'  {alg+tag:<18} {lat:>10.2f} {eng:>10.4f} '
              f'{prv:>8.4f} {sla:>8.2f} {thr:>10.1f}')


def _save_table3_csv(mc_summary, alg_names, path):
    rows = []
    scale = mc_summary.get(PRIMARY_SCALE, {})
    for alg in alg_names:
        d = scale.get(alg, {})
        rows.append({
            'algorithm': alg,
            'avg_latency_ms_mean':   d.get('avg_latency_ms',    {}).get('mean', 0),
            'avg_latency_ms_std':    d.get('avg_latency_ms',    {}).get('std',  0),
            'avg_energy_mj_mean':    d.get('avg_energy_mj',     {}).get('mean', 0),
            'avg_energy_mj_std':     d.get('avg_energy_mj',     {}).get('std',  0),
            'avg_privacy_risk_mean': d.get('avg_privacy_risk',  {}).get('mean', 0),
            'avg_privacy_risk_std':  d.get('avg_privacy_risk',  {}).get('std',  0),
            'sla_violation_pct_mean':d.get('sla_violation_pct', {}).get('mean', 0),
            'sla_violation_pct_std': d.get('sla_violation_pct', {}).get('std',  0),
            'n':                     d.get('avg_latency_ms',    {}).get('n',    0),
        })
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f'[Q1-MC] Saved {path}')


def _save_json(obj, path):
    def _conv(o):
        if isinstance(o, defaultdict):
            return {k: _conv(v) for k, v in o.items()}
        if isinstance(o, dict):
            return {str(k): _conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_conv(x) for x in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(_conv(obj), fh, indent=2)
    print(f'[Q1-MC] Saved {path}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Q1 master Monte Carlo driver (Fixes 1, 2, 3, 5, 7).'
    )
    parser.add_argument('--n_runs', type=int, default=N_RUNS,
                        help=f'Independent runs per cell (default {N_RUNS})')
    parser.add_argument('--scales', type=int, nargs='+', default=None,
                        help=f'Task scales (default {TASK_SCALES})')
    parser.add_argument('--output', type=str, default=None,
                        help='Override results directory')
    parser.add_argument('--quick', action='store_true',
                        help='5 runs only — debug, not for publication')
    parser.add_argument('--registry', choices=['main', 'experiments', 'all'],
                        default='main',
                        help='main = 9-algorithm comparison; experiments = '
                             'E1-E3 decomposition arms (K-sweep, Random-K, '
                             'q-mixed, Static-Tier, lambda_P)')
    parser.add_argument('--algorithms', nargs='+', default=None,
                        help='Restrict to these registry names')
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--no-raw', action='store_true',
                        help='Do not write per-run raw logs under results/raw/')
    parser.add_argument('--n-fog', type=int, default=N_FOG_NODES,
                        help='Fog-node count M (E4; action set = M + 2)')
    parser.add_argument('--ecg-payload-bits', type=int, default=None,
                        help='Override ECG D_i (sensitivity; config '
                             'ECG_PAYLOAD_SENSITIVITY_BITS = 80000 = 10 KB)')
    parser.add_argument('--arrival', choices=['poisson', 'mmpp2'],
                        default='poisson', help='Arrival process (E7)')
    parser.add_argument('--load-rho', type=float, default=None,
                        help='Target offered edge utilisation (E7), sets the '
                             'absolute arrival rate')
    parser.add_argument('--ci-noise', type=float, default=None,
                        help='Gaussian sigma on the scheduler-visible CI (E14)')
    parser.add_argument('--ci-misclass', type=float, default=None,
                        help='CI tier misclassification probability (E14)')
    parser.add_argument('--warm-start', type=int, default=None,
                        help='Pre-train DQN arms on N_pre tasks (E12)')
    parser.add_argument('--result-size-bits', type=int, default=None,
                        help='Override result size S_res (sensitivity; '
                             'config RESULT_SIZE_SENSITIVITY_BITS)')
    args = parser.parse_args()

    n_runs = 5 if args.quick else args.n_runs
    scales = args.scales if args.scales is not None else TASK_SCALES

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    overrides = {'ecg_payload_bits': args.ecg_payload_bits,
                 'result_size_bits': args.result_size_bits,
                 'arrival_process': (args.arrival if args.arrival != 'poisson'
                                     else None),
                 'load_rho': args.load_rho,
                 'warm_start_tasks': args.warm_start,
                 'ci_noise_sigma': args.ci_noise,
                 'ci_misclass_prob': args.ci_misclass}
    overrides = {k: v for k, v in overrides.items() if v is not None} or None
    results_dir = (Path(args.output) if args.output
                   else project_root / 'results')
    if overrides and not args.output:
        # Never let a sensitivity run overwrite the main-model files
        results_dir = results_dir / 'sensitivity' / overrides_tag(overrides)
    if args.n_fog != N_FOG_NODES and not args.output:
        results_dir = results_dir / 'scaling' / f'M{args.n_fog}'

    from src.analysis.manifest import Manifest
    with Manifest(results_dir, f'cli_run_full_{args.registry}', vars(args),
                  n_runs=n_runs, scales=scales):
        _run_full_cli(args, scales, n_runs, results_dir, overrides)


def _run_full_cli(args, scales, n_runs, results_dir, overrides):
    run_full(scales, n_runs, results_dir, workers=args.workers,
             registry_name=args.registry, algorithms=args.algorithms,
             raw_logs=not args.no_raw, task_overrides=overrides,
             n_fog_nodes=args.n_fog)


if __name__ == '__main__':
    main()
