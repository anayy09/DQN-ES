"""
scheduling_overhead.py — Fix C (Fix2.md): wall-clock scheduling latency.

Instruments the per-task scheduling decision for DQN-ES, PSO+DQN, and
ES-only at N=1000 over 30 Monte Carlo runs.

Timing definition
-----------------
For DQN-ES and PSO+DQN: from state observation (line 1 of Algorithm 1) to
dispatch decision (line 15 = just after the inner-search returns the node_id),
EXCLUDING the Bellman update (lines 16–18).  This is captured by
dispatch_times_ms in each scheduler.

For ES-only: dispatch_times_ms captures the full BBO search call (no DQN or
Bellman update).

Note: the environment also records `scheduling_overhead_ms` which covers the
FULL select_node() call (including Bellman update for DRL methods).  Both are
reported so the reader can assess how much the training step adds.

Outputs
-------
results/scheduling_overhead.csv
  columns: algorithm, run, task_idx, dispatch_time_ms, full_select_time_ms

results/scheduling_overhead_summary.csv
  columns: algorithm, mean_ms, median_ms, p95_ms, max_ms
  (dispatch times, excluding Bellman update)
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, quantiles
from typing import List

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.config import (
    GLOBAL_SEED,
    N_FOG_NODES,
    N_WEARABLES,
    PRIMARY_SCALE,
)
from src.simulation.replicate import build_synthetic_replicate, run_scheduler

# Algorithms timed per Fix C spec
TIMED_ALGORITHMS = ['DQN-ES', 'ES-only']


# ---------------------------------------------------------------------------
# Worker (top-level for spawn pickling)
# ---------------------------------------------------------------------------

def _run_cell(payload: tuple) -> tuple:
    """
    payload = (alg_name, run_id, n_tasks)
    Returns (alg_name, run_id, dispatch_times_list, full_select_times_list).
    """
    alg_name, run_id, n_tasks = payload

    from src.config import get_full_algorithm_registry
    registry = get_full_algorithm_registry()
    if alg_name not in registry:
        return alg_name, run_id, [], []

    seeds, topo, tasks = build_synthetic_replicate(run_id, n_tasks, 'mixed')
    results, sched = run_scheduler(registry[alg_name], topo, tasks, seeds)

    # Dispatch times (excluding Bellman update) from scheduler attribute
    dispatch_times = list(getattr(sched, 'dispatch_times_ms', []))

    # Full select_node() times from environment step results
    full_select_times = [r.get('scheduling_overhead_ms', 0.0) for r in results]

    return alg_name, run_id, dispatch_times, full_select_times


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_overhead_analysis(
    results_dir: Path,
    n_runs: int = 30,
    n_tasks: int = PRIMARY_SCALE,
    workers: int | None = None,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    payloads = [
        (alg, rid, n_tasks)
        for alg in TIMED_ALGORITHMS
        for rid in range(n_runs)
    ]
    n_jobs = len(payloads)

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, n_jobs)

    print(f'[OVERHEAD] Dispatching {n_jobs} cells over {workers} workers '
          f'(algorithms={TIMED_ALGORITHMS}, n_runs={n_runs}, N={n_tasks})')

    raw_dispatch:  dict = defaultdict(list)   # alg -> list of per-run lists
    raw_full:      dict = defaultdict(list)
    completed = 0
    t0 = time.time()

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=workers) as pool:
        for alg, rid, disp_times, full_times in pool.imap_unordered(
                _run_cell, payloads, chunksize=1):
            completed += 1
            if disp_times:
                raw_dispatch[alg].append({'run': rid, 'times': disp_times})
            if full_times:
                raw_full[alg].append({'run': rid, 'times': full_times})
            elapsed = time.time() - t0
            print(f'  [{completed:3d}/{n_jobs}] {alg:<10s} run={rid}'
                  f'  dispatch_n={len(disp_times)}  wall={elapsed:.1f}s',
                  flush=True)

    print(f'\n[OVERHEAD] Done in {time.time()-t0:.1f}s')

    # Write per-task CSV
    csv_path = results_dir / 'scheduling_overhead.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=[
            'algorithm', 'run', 'task_idx',
            'dispatch_time_ms', 'full_select_time_ms',
        ])
        wr.writeheader()
        for alg in TIMED_ALGORITHMS:
            d_runs = {r['run']: r['times'] for r in raw_dispatch[alg]}
            f_runs = {r['run']: r['times'] for r in raw_full[alg]}
            for rid in range(n_runs):
                d_times = d_runs.get(rid, [])
                f_times = f_runs.get(rid, [])
                n_t = max(len(d_times), len(f_times))
                for ti in range(n_t):
                    wr.writerow({
                        'algorithm':          alg,
                        'run':                rid,
                        'task_idx':           ti,
                        'dispatch_time_ms':   d_times[ti] if ti < len(d_times) else '',
                        'full_select_time_ms': f_times[ti] if ti < len(f_times) else '',
                    })
    print(f'[OVERHEAD] Saved {csv_path}')

    # Summary statistics
    summary_rows = []
    summary = {}
    for alg in TIMED_ALGORITHMS:
        all_disp = [t for r in raw_dispatch[alg] for t in r['times']]
        if not all_disp:
            continue
        all_disp_arr = sorted(all_disp)
        n = len(all_disp_arr)
        mu   = float(np.mean(all_disp_arr))
        med  = float(np.median(all_disp_arr))
        p95  = float(all_disp_arr[int(0.95 * n)])
        vmax = float(all_disp_arr[-1])
        summary[alg] = {'mean_ms': mu, 'median_ms': med, 'p95_ms': p95, 'max_ms': vmax}
        summary_rows.append({
            'algorithm': alg,
            'mean_ms': f'{mu:.4f}',
            'median_ms': f'{med:.4f}',
            'p95_ms': f'{p95:.4f}',
            'max_ms': f'{vmax:.4f}',
        })

    sum_path = results_dir / 'scheduling_overhead_summary.csv'
    if summary_rows:
        with open(sum_path, 'w', newline='', encoding='utf-8') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            wr.writeheader()
            wr.writerows(summary_rows)

    print('\n[OVERHEAD] Per-task dispatch time (excl. Bellman update):')
    print(f'  {"Algorithm":<12s} {"Mean":>10s} {"Median":>10s} '
          f'{"p95":>10s} {"Max":>10s}')
    for row in summary_rows:
        print(f'  {row["algorithm"]:<12s} {row["mean_ms"]:>10s} '
              f'{row["median_ms"]:>10s} {row["p95_ms"]:>10s} '
              f'{row["max_ms"]:>10s}')
    print(f'[OVERHEAD] Saved {sum_path}')
    return summary


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, default=None)
    p.add_argument('--n_runs',      type=int, default=30)
    p.add_argument('--n_tasks',     type=int, default=PRIMARY_SCALE)
    p.add_argument('--workers',     type=int, default=None)
    args = p.parse_args()

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    results_dir  = Path(args.results_dir) if args.results_dir \
        else project_root / 'results'

    run_overhead_analysis(results_dir, n_runs=args.n_runs,
                          n_tasks=args.n_tasks, workers=args.workers)


if __name__ == '__main__':
    mp.freeze_support()
    main()
