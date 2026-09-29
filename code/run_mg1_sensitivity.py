"""
run_mg1_sensitivity.py — M/G/1 queueing sensitivity analysis (parallelised).

Replaces the M/M/1 queue delay (c_v=1) with M/G/1 for c_v in {1.5, 2.5}
using the Pollaczek–Khinchine formula:
    E[W_G] = E[W_MM1] * (1 + c_v^2) / 2

Runs all algorithms at N=1000 for 30 replicates under each c_v setting.
Optimized to simulate each run exactly once, then applying the latency scaling for each c_v in post-processing.
Parallelized over individual runs instead of cv/alg pairs.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PROJECT_ROOT = Path(BASE).parent
RESULTS_DIR  = PROJECT_ROOT / 'results'

from src.config import GLOBAL_SEED, N_FOG_NODES, N_RUNS, N_WEARABLES, get_full_algorithm_registry
from src.simulation.topology import build_healthcare_topology

N_TASKS  = 1000
CV_VALUES = [1.0, 1.5, 2.5]
N_RUNS_CV = N_RUNS   # 30
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)


# ---------------------------------------------------------------------------
# Top-level worker — must be importable (no closures) for multiprocessing
# ---------------------------------------------------------------------------

def _run_single_worker(args: tuple):
    """Run one replicate for one alg_name, and compute metrics for all cv values."""
    run_id, alg_name, n_tasks, seed_base, base_path = args

    if base_path not in sys.path:
        sys.path.insert(0, base_path)

    import random as _r
    import numpy as _np
    import warnings as _w

    from src.config import get_full_algorithm_registry
    from src.simulation.replicate import build_synthetic_replicate, run_scheduler

    sched_cls = get_full_algorithm_registry()[alg_name]

    try:
        seeds, topo, tasks = build_synthetic_replicate(run_id, n_tasks, 'mixed')
        res, _ = run_scheduler(sched_cls, topo, tasks, seeds)

        if not res:
            return run_id, alg_name, {}

        q_frac = 0.25
        dl = [t.max_delay_s * 1000.0 for t in tasks]
        eng = float(_np.mean([r['energy_mj'] for r in res]))
        priv = float(_np.mean([r['privacy_risk'] for r in res]))

        cv_metrics = {}
        for cv in CV_VALUES:
            scale  = (1.0 + cv ** 2) / 2.0
            lats = []
            for r in res:
                lat = r['latency_ms']
                if r.get('node_type', '') != 'local':
                    lat = lat * (1 - q_frac) + lat * q_frac * scale
                lats.append(lat)
            
            sla  = 100.0 * sum(1 for l, d in zip(lats, dl) if l > d) / len(res)
            
            cv_metrics[cv] = {
                'avg_latency_ms': float(_np.mean(lats)),
                'avg_energy_mj': eng,
                'avg_privacy_risk': priv,
                'sla_violation_pct': sla,
            }
        
        return run_id, alg_name, cv_metrics

    except Exception as exc:
        _w.warn(f'[M/G/1] {alg_name} run={run_id}: {exc}')
        return run_id, alg_name, {}


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def _agg(runs: list[dict], key: str) -> dict:
    vals = np.array([r[key] for r in runs], dtype=float)
    return {'mean': float(vals.mean()), 'std': float(vals.std()),
            'min': float(vals.min()),   'max': float(vals.max()),
            'n': int(len(vals))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    registry    = get_full_algorithm_registry()
    algs_to_test = [k for k in registry if k not in ('Local-Only', 'Cloud-Only')]

    print(f'[M/G/1 OPTIMIZED] {len(algs_to_test)} algorithms × {N_RUNS_CV} runs = {len(algs_to_test)*N_RUNS_CV} total simulations')
    print(f'[M/G/1 OPTIMIZED] Workers: {N_WORKERS}')
    print(f'[M/G/1 OPTIMIZED] Algorithms: {algs_to_test}', flush=True)

    job_args = [
        (run_id, alg, N_TASKS, GLOBAL_SEED, BASE)
        for alg in algs_to_test
        for run_id in range(N_RUNS_CV)
    ]

    # raw[cv][alg] = [run_metrics, ...]
    raw: dict[float, dict[str, list]] = {cv: {alg: [] for alg in algs_to_test} for cv in CV_VALUES}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_single_worker, a): a for a in job_args}
        completed = 0
        total = len(job_args)
        for fut in as_completed(futures):
            completed += 1
            try:
                run_id, alg_name, cv_metrics = fut.result()
                if cv_metrics:
                    for cv in CV_VALUES:
                        raw[cv][alg_name].append(cv_metrics[cv])
            except Exception as exc:
                run_id, alg_name, *_ = futures[fut]
                print(f'  [ERROR] {alg_name} run={run_id}: {exc}', flush=True)
            
            if completed % 10 == 0 or completed == total:
                print(f'  [Progress] {completed}/{total} runs completed.', flush=True)

    # Aggregate
    results: dict[float, dict] = {cv: {} for cv in CV_VALUES}
    for cv in CV_VALUES:
        for alg in algs_to_test:
            runs = raw[cv].get(alg, [])
            if not runs:
                results[cv][alg] = {}
                continue
            results[cv][alg] = {
                'avg_latency_ms':    _agg(runs, 'avg_latency_ms'),
                'avg_energy_mj':     _agg(runs, 'avg_energy_mj'),
                'avg_privacy_risk':  _agg(runs, 'avg_privacy_risk'),
                'sla_violation_pct': _agg(runs, 'sla_violation_pct'),
            }

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / 'mg1_sensitivity.json'
    with open(out_path, 'w') as fh:
        json.dump({str(cv): results[cv] for cv in CV_VALUES}, fh, indent=2)
    print(f'\n[SAVE] {out_path}')

    # Save CSV
    csv_path = RESULTS_DIR / 'table_mg1.csv'
    header = (['Algorithm']
              + [f'cv={cv} lat±std' for cv in CV_VALUES]
              + [f'cv={cv} priv'         for cv in CV_VALUES]
              + [f'cv={cv} SLA%'         for cv in CV_VALUES])
    rows = [header]
    for alg in algs_to_test:
        row = [alg]
        for cv in CV_VALUES:
            d = results[cv].get(alg, {})
            lat = d.get('avg_latency_ms', {})
            row.append(f'{lat.get("mean",0):.2f}±{lat.get("std",0):.2f}')
        for cv in CV_VALUES:
            d = results[cv].get(alg, {})
            row.append(f'{d.get("avg_privacy_risk",{}).get("mean",0):.4f}')
        for cv in CV_VALUES:
            d = results[cv].get(alg, {})
            row.append(f'{d.get("sla_violation_pct",{}).get("mean",0):.2f}')
        rows.append(row)

    with open(csv_path, 'w', newline='') as fh:
        csv.writer(fh).writerows(rows)
    print(f'[SAVE] {csv_path}')

    # Ordering check
    print('\n[ORDERING CHECK] Privacy risk ordering under M/G/1:')
    for cv in CV_VALUES:
        privs = {alg: results[cv].get(alg, {}).get('avg_privacy_risk', {}).get('mean', 999)
                 for alg in algs_to_test}
        ordered = sorted(privs, key=privs.get)
        print(f'  cv={cv}: {" < ".join(f"{a}={privs[a]:.4f}" for a in ordered)}')

    print('\nDone.')


if __name__ == '__main__':
    main()
