"""
latency_decomposition.py — latency component breakdown.

Decomposes mean end-to-end latency at N=1000 for config.ROBUSTNESS_ARMS
(G1-4: DQN-ES, ES-only, Random-K[K=3], q-mixed[q=0.5], Static-Tier) into
  (1) transmission latency  t_tx   = D_i / R_{i,j}
  (2) queuing delay         t_queue (FIFO wait, D16)
  (3) compute at destination t_proc = C_i / f_j
  (4) propagation and (5) result download (E9)

Outputs
-------
results/latency_decomposition.csv
  columns: algorithm, N, mean_tx_ms, mean_queue_ms, mean_compute_ms,
           mean_prop_ms, mean_downlink_ms, mean_total_ms
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

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
    ROBUSTNESS_ARMS,
)
from src.simulation.replicate import build_synthetic_replicate, run_scheduler

# Algorithms for decomposition: the robustness arms (G1-4)
DECOMP_ALGORITHMS = list(ROBUSTNESS_ARMS)


def _run_cell(payload: tuple) -> tuple:
    """payload = (alg_name, run_id, n_tasks)"""
    alg_name, run_id, n_tasks = payload

    from src.config import get_experiment_registry, get_full_algorithm_registry
    registry = {**get_full_algorithm_registry(), **get_experiment_registry()}
    if alg_name not in registry:
        return alg_name, run_id, None

    seeds, topo, tasks = build_synthetic_replicate(run_id, n_tasks, 'mixed')
    results, _ = run_scheduler(registry[alg_name], topo, tasks, seeds)
    if not results:
        return alg_name, run_id, None

    metrics = {
        'mean_tx_ms':      mean(r['latency_tx_ms']      for r in results),
        'mean_queue_ms':   mean(r['latency_queue_ms']    for r in results),
        'mean_compute_ms': mean(r['latency_compute_ms']  for r in results),
        'mean_prop_ms':    mean(r['latency_prop_ms']     for r in results),
        'mean_downlink_ms': mean(r['latency_downlink_ms'] for r in results),
        'mean_total_ms':   mean(r['latency_ms']          for r in results),
    }
    return alg_name, run_id, metrics


def run_decomposition(
    results_dir: Path,
    n_runs: int = 30,
    n_tasks: int = PRIMARY_SCALE,
    workers: int | None = None,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    payloads = [
        (alg, rid, n_tasks)
        for alg in DECOMP_ALGORITHMS
        for rid in range(n_runs)
    ]
    n_jobs = len(payloads)

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, n_jobs)

    print(f'[DECOMP] {n_jobs} cells, {workers} workers, N={n_tasks}')

    raw: dict = defaultdict(list)
    t0 = time.time()

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=workers) as pool:
        for alg, rid, metrics in pool.imap_unordered(_run_cell, payloads):
            if metrics is not None:
                raw[alg].append(metrics)
            print(f'  {alg} run={rid}  wall={time.time()-t0:.1f}s', flush=True)

    comp_keys = ['mean_tx_ms', 'mean_queue_ms', 'mean_compute_ms',
                 'mean_prop_ms', 'mean_downlink_ms', 'mean_total_ms']
    summary = {}
    csv_rows = []
    for alg in DECOMP_ALGORITHMS:
        runs = raw[alg]
        if not runs:
            continue
        row = {'algorithm': alg, 'N': n_tasks}
        agg = {}
        for k in comp_keys:
            vs = np.array([r[k] for r in runs], dtype=float)
            agg[k] = {'mean': float(vs.mean()), 'std': float(vs.std())}
            row[k] = f'{vs.mean():.4f} ± {vs.std():.4f}'
        summary[alg] = agg
        row_flat = {'algorithm': alg, 'N': n_tasks}
        for k in comp_keys:
            row_flat[k] = float(np.mean([r[k] for r in runs]))
        csv_rows.append(row_flat)

    csv_path = results_dir / 'latency_decomposition.csv'
    if csv_rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
            wr.writeheader()
            wr.writerows(csv_rows)

    print(f'\n[DECOMP] Latency breakdown at N={n_tasks}:')
    print(f'  {"Algorithm":<12s} {"Tx":>8s} {"Queue":>8s} '
          f'{"Compute":>10s} {"Prop":>8s} {"Total":>8s}')
    for alg in DECOMP_ALGORITHMS:
        if alg not in summary:
            continue
        s = summary[alg]
        print(f'  {alg:<12s} '
              f'{s["mean_tx_ms"]["mean"]:>8.2f} '
              f'{s["mean_queue_ms"]["mean"]:>8.2f} '
              f'{s["mean_compute_ms"]["mean"]:>10.2f} '
              f'{s["mean_prop_ms"]["mean"]:>8.2f} '
              f'{s["mean_total_ms"]["mean"]:>8.2f}')
    print(f'[DECOMP] Saved {csv_path}')
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

    run_decomposition(results_dir, n_runs=args.n_runs,
                      n_tasks=args.n_tasks, workers=args.workers)


if __name__ == '__main__':
    mp.freeze_support()
    main()
