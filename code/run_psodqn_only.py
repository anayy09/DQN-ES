"""
run_psodqn_only.py — Run PSO+DQN Monte Carlo and merge into existing results.

Runs only PSO+DQN (30 runs × 5 scales), then patches mc_full_summary.json
and mc_full_results.json in-place so existing algorithm results are preserved.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PROJECT_ROOT = Path(BASE).parent
RESULTS_DIR  = PROJECT_ROOT / 'results'

from src.config import GLOBAL_SEED, N_FOG_NODES, N_RUNS, N_WEARABLES, TASK_SCALES
from src.algorithms.pso_dqn import PSODQNScheduler
from src.core.task import HealthcareTask
from src.data_ingestion.event_generator import generate_synthetic_tasks
from src.simulation.environment import OffloadingEnvironment
from src.simulation.replicate import build_synthetic_replicate, run_scheduler

ALG_NAME = 'PSO+DQN'

def _run_single(n_tasks: int, run_id: int) -> dict:
    seeds, topo, tasks = build_synthetic_replicate(run_id, n_tasks, 'mixed')
    results, _ = run_scheduler(PSODQNScheduler, topo, tasks, seeds)

    if not results:
        return {'avg_latency_ms': 0.0, 'avg_energy_mj': 0.0,
                'avg_privacy_risk': 0.0, 'sla_violation_pct': 0.0,
                'throughput': 0.0}

    timestamps = [r['timestamp'] for r in results]
    span = max(timestamps) - min(timestamps) + 1e-6
    return {
        'avg_latency_ms':    mean(r['latency_ms']    for r in results),
        'avg_energy_mj':     mean(r['energy_mj']     for r in results),
        'avg_privacy_risk':  mean(r['privacy_risk']  for r in results),
        'sla_violation_pct': 100.0 * sum(r['sla_violated'] for r in results) / len(results),
        'throughput':        n_tasks / span,
    }


def main():
    metric_keys = ['avg_latency_ms', 'avg_energy_mj',
                   'avg_privacy_risk', 'sla_violation_pct', 'throughput']

    raw_by_scale:  dict[int, list[dict]] = {}
    summ_by_scale: dict[int, dict]       = {}

    t0 = time.time()
    for n_tasks in TASK_SCALES:
        print(f'\n[PSO+DQN] n={n_tasks} — {N_RUNS} runs', flush=True)
        runs = []
        for run_id in range(N_RUNS):
            try:
                m = _run_single(n_tasks, run_id)
                runs.append(m)
                if (run_id + 1) % 5 == 0:
                    pr = np.mean([r['avg_privacy_risk'] for r in runs])
                    lt = np.mean([r['avg_latency_ms']   for r in runs])
                    print(f'  run {run_id+1:2d}/{N_RUNS}  lat={lt:.2f}ms  priv={pr:.4f}', flush=True)
            except Exception as exc:
                warnings.warn(f'[PSO+DQN] n={n_tasks} run={run_id} failed: {exc}')

        raw_by_scale[n_tasks] = runs
        agg = {}
        for key in metric_keys:
            vals = np.array([r[key] for r in runs if key in r], dtype=float)
            agg[key] = {
                'mean': float(vals.mean()), 'std': float(vals.std()),
                'min': float(vals.min()),   'max': float(vals.max()),
                'n': int(len(vals)),        'samples': [float(v) for v in vals],
            }
        summ_by_scale[n_tasks] = agg
        print(f'  SUMMARY  lat={agg["avg_latency_ms"]["mean"]:.2f}±{agg["avg_latency_ms"]["std"]:.2f}  '
              f'eng={agg["avg_energy_mj"]["mean"]:.4f}±{agg["avg_energy_mj"]["std"]:.4f}  '
              f'priv={agg["avg_privacy_risk"]["mean"]:.4f}±{agg["avg_privacy_risk"]["std"]:.4f}  '
              f'sla={agg["sla_violation_pct"]["mean"]:.2f}%', flush=True)

    elapsed = time.time() - t0
    print(f'\n[PSO+DQN] Done in {elapsed:.1f}s')

    # --- Merge into mc_full_summary.json ---
    summ_path = RESULTS_DIR / 'mc_full_summary.json'
    if summ_path.exists():
        with open(summ_path, 'r') as fh:
            existing_summary = json.load(fh)
    else:
        existing_summary = {}

    for n_tasks in TASK_SCALES:
        key = str(n_tasks)
        if key not in existing_summary:
            existing_summary[key] = {}
        existing_summary[key][ALG_NAME] = summ_by_scale[n_tasks]

    with open(summ_path, 'w') as fh:
        json.dump(existing_summary, fh, indent=2)
    print(f'[MERGE] Updated {summ_path}')

    # --- Merge into mc_full_results.json ---
    raw_path = RESULTS_DIR / 'mc_full_results.json'
    if raw_path.exists():
        with open(raw_path, 'r') as fh:
            existing_raw = json.load(fh)
    else:
        existing_raw = {}

    for n_tasks in TASK_SCALES:
        key = str(n_tasks)
        if key not in existing_raw:
            existing_raw[key] = {}
        existing_raw[key][ALG_NAME] = raw_by_scale[n_tasks]

    with open(raw_path, 'w') as fh:
        json.dump(existing_raw, fh, indent=2)
    print(f'[MERGE] Updated {raw_path}')

    # Print framing assessment
    n1000 = summ_by_scale.get(1000, {})
    bbo_summ_path = RESULTS_DIR / 'mc_full_summary.json'
    with open(bbo_summ_path) as fh:
        full = json.load(fh)
    bbo_priv = full['1000']['DQN-ES']['avg_privacy_risk']['mean']
    pso_priv = n1000['avg_privacy_risk']['mean']
    diff_pct = 100.0 * (pso_priv - bbo_priv) / max(abs(bbo_priv), 1e-9)
    print(f'\n[FRAMING] DQN-ES privacy={bbo_priv:.4f}  PSO+DQN privacy={pso_priv:.4f}  diff={diff_pct:+.1f}%')
    if bbo_priv < pso_priv * 0.95:
        print('  → BBO inner loop provides measurable privacy benefit over PSO inner loop.')
        print('    Central attribution claim is supported.')
    elif abs(diff_pct) < 5.0:
        print('  → DQN-ES and PSO+DQN are within 5% on privacy.')
        print('    REFRAME: contribution is DRL+bio-inspired coupling, not BBO specifically.')
    else:
        print('  → Mixed — review metric-by-metric before finalising framing.')


if __name__ == '__main__':
    main()
