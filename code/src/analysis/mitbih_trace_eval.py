"""
mitbih_trace_eval.py — Fix 10: MIT-BIH real-trace evaluation, parallel.

Each (algorithm, run_id) cell is fully independent, so the n_runs * n_algs
matrix is dispatched to a multiprocessing.Pool.  On an 8-core laptop this
typically delivers a 5-7x wall-clock speedup over the serial driver.

The trace itself (8 640 windows) is pre-cached on disk via the existing
parse_mitbih ingestion and loaded once per worker via a Pool initializer
so we don't re-pickle 10 K events for every job.

Outputs:
  results/table5_mitbih_trace.csv
  results/mitbih_trace_raw.json
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import List

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.config import (
    GLOBAL_SEED,
    MITBIH_DEADLINE_S,
    MITBIH_N_RUNS,
    MITBIH_PAYLOAD_BITS,
    MITBIH_RHO,
    N_FOG_NODES,
    N_WEARABLES,
    get_full_algorithm_registry,
    get_registry,
    replicate_seeds,
    seed_global_rngs,
)
from src.core.task import HealthcareTask, TASK_PROFILES
from src.data_ingestion.parse_mitbih import load_mitbih_events
from src.simulation.environment import OffloadingEnvironment
from src.simulation.episode_log import add_steady_state, raw_log_path, write_raw_log
from src.simulation.replicate import build_topology, run_scheduler


# =========================================================================
# Worker-side globals  (populated by _worker_init on each child process)
# =========================================================================
_WORKER_EVENTS: list | None = None


def _worker_init(events_list: list) -> None:
    """Pool initializer: every worker receives the event list exactly once."""
    global _WORKER_EVENTS
    _WORKER_EVENTS = events_list


def _events_to_tasks(events: list, topo, rng) -> List[HealthcareTask]:
    """Convert MIT-BIH window events to a HealthcareTask trace."""
    wids = [nid for nid, n in topo.nodes.items()
            if n.node_type == 'wearable']
    tasks = []
    t_now = 0.0
    cpu_cycles = TASK_PROFILES['ecg_analysis']['cpu_cycles']
    for i, ev in enumerate(events):
        t_now += rng.expovariate(20.0)
        dev = wids[i % len(wids)]
        tasks.append(HealthcareTask(
            task_id=i,
            device_id=dev,
            timestamp=t_now,
            data_size_bits=MITBIH_PAYLOAD_BITS,
            cpu_cycles=cpu_cycles,
            max_delay_s=MITBIH_DEADLINE_S,
            privacy_sensitivity=MITBIH_RHO,
            ci_score=float(ev.get('ci_score', 0.5)),
            attack_probability=0.0,
            source='mitbih',
            task_type='ecg_analysis',
        ))
    return tasks


# =========================================================================
# Worker entry point  (must be at module top level for Windows pickling)
# =========================================================================
def _run_cell(payload: tuple) -> tuple:
    """
    Run one (algorithm, run_id) cell.

    payload = (alg_name, run_id)
    Returns (alg_name, run_id, metrics_dict_or_None, elapsed_seconds).
    """
    import random as _r
    alg_name, run_id, raw_dir = payload
    t0 = time.time()

    seeds = replicate_seeds(run_id, 0)
    seed_global_rngs(seeds['base'])
    local_rng = _r.Random(seeds['task'])

    topo = build_topology(run_id, 0)
    tasks = _events_to_tasks(_WORKER_EVENTS, topo, local_rng)

    registry = get_registry('all')
    res, _ = run_scheduler(registry[alg_name], topo, tasks, seeds)

    if not res:
        return alg_name, run_id, None, time.time() - t0
    if raw_dir is not None:
        write_raw_log(res, raw_log_path(raw_dir, len(tasks), alg_name, run_id),
                      alg_name, run_id, len(tasks))

    metrics = {
        'avg_latency_ms':    mean(r['latency_ms']    for r in res),
        'avg_energy_mj':     mean(r['energy_mj']     for r in res),
        'avg_privacy_risk':  mean(r['privacy_risk']  for r in res),
        'sla_violation_pct': 100.0 * sum(r['sla_violated']
                                         for r in res) / len(res),
    }
    add_steady_state(metrics, res)
    return alg_name, run_id, metrics, time.time() - t0


# =========================================================================
# Driver
# =========================================================================
def run_mitbih_trace(
    data_dir: Path,
    results_dir: Path,
    n_runs: int = MITBIH_N_RUNS,
    workers: int | None = None,
    smoke: bool = False,
    max_tasks: int | None = None,
    algorithms: list | None = None,
    raw_logs: bool = True,
) -> dict:
    """
    Run the MIT-BIH trace evaluation in parallel.

    Parameters
    ----------
    data_dir : Path
        Project data root (contains MIT-BIH-Arrhythmia/).
    results_dir : Path
        Where to write CSV and JSON outputs.
    n_runs : int
        Independent Monte Carlo replicates per algorithm.
    workers : int or None
        Number of worker processes; defaults to os.cpu_count() - 1.
    smoke : bool
        If True, run just one (alg, run_id) cell per algorithm to verify
        the parallel pipeline end-to-end without launching the full sweep.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f'[MIT-BIH] Loading events from {data_dir} ...')
    events = load_mitbih_events(str(data_dir))
    print(f'[MIT-BIH] Loaded {len(events)} window events.')

    registry = get_full_algorithm_registry()
    alg_names = list(algorithms) if algorithms else list(registry.keys())
    unknown = [a for a in alg_names if a not in get_registry('all')]
    if unknown:
        raise KeyError(f'unknown algorithms: {unknown}')

    if smoke:
        n_runs = 1
        if max_tasks is None:
            max_tasks = 200
        print(f'[MIT-BIH] SMOKE MODE: 1 run per algorithm on first '
              f'{max_tasks} tasks.')

    if max_tasks is not None and max_tasks < len(events):
        events = events[:max_tasks]
        print(f'[MIT-BIH] Truncated to {len(events)} tasks (max_tasks).')

    raw_dir = (results_dir / 'raw' / 'mitbih') if raw_logs else None
    payloads = [(alg, rid, raw_dir) for alg in alg_names for rid in range(n_runs)]
    n_jobs = len(payloads)

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, n_jobs)

    print(f'[MIT-BIH] Dispatching {n_jobs} cells over {workers} workers '
          f'(algorithms={len(alg_names)}, n_runs={n_runs}).')

    raw: dict = defaultdict(list)
    t_start = time.time()
    completed = 0

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=workers, initializer=_worker_init,
                  initargs=(events,)) as pool:
        # imap_unordered: results stream back as soon as each finishes
        for alg, rid, metrics, dt in pool.imap_unordered(
                _run_cell, payloads, chunksize=1):
            completed += 1
            if metrics is not None:
                raw[alg].append(metrics)
            wall = time.time() - t_start
            eta = wall / completed * (n_jobs - completed) if completed else 0
            print(f'  [{completed:3d}/{n_jobs}] {alg:<12s} run={rid:<3d} '
                  f'dt={dt:6.1f}s   wall={wall/60:5.1f}m   '
                  f'eta={eta/60:5.1f}m',
                  flush=True)

    elapsed = time.time() - t_start
    print(f'\n[MIT-BIH] Wall time: {elapsed/60:.1f} min '
          f'({elapsed:.0f} s) for {n_jobs} cells.')

    # ---------- Aggregate ----------
    summary = {}
    csv_rows = []
    for alg in alg_names:
        runs = raw[alg]
        row = {'algorithm': alg}
        agg = {}
        for k in ['avg_latency_ms', 'avg_energy_mj',
                  'avg_privacy_risk', 'sla_violation_pct']:
            vs = np.array([r[k] for r in runs], dtype=float)
            if len(vs) == 0:
                agg[k] = {'mean': 0.0, 'std': 0.0, 'samples': []}
                row[f'{k}_mean'] = 0.0
                row[f'{k}_std']  = 0.0
                continue
            agg[k] = {'mean': float(vs.mean()),
                      'std':  float(vs.std()),
                      'samples': vs.tolist()}
            row[f'{k}_mean'] = float(vs.mean())
            row[f'{k}_std']  = float(vs.std())
        # Steady-state R_P (F7): only runs long enough to pass warm-up
        ss = np.array([r['avg_privacy_risk_ss'] for r in runs
                       if 'avg_privacy_risk_ss' in r], dtype=float)
        if len(ss):
            agg['avg_privacy_risk_ss'] = {'mean': float(ss.mean()),
                                          'std': float(ss.std()),
                                          'samples': ss.tolist()}
            row['avg_privacy_risk_ss_mean'] = float(ss.mean())
            row['avg_privacy_risk_ss_std'] = float(ss.std())
        summary[alg] = agg
        csv_rows.append(row)

    suffix = '_smoke' if smoke else ''
    csv_path = results_dir / f'table5_mitbih_trace{suffix}.csv'
    json_path = results_dir / f'mitbih_trace_raw{suffix}.json'
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        fields = list(dict.fromkeys(k for r in csv_rows for k in r))
        wr = csv.DictWriter(fh, fieldnames=fields, restval='')
        wr.writeheader()
        wr.writerows(csv_rows)
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)

    print(f'\n[MIT-BIH] Real-trace results '
          f'(n_tasks={len(events)}, runs={n_runs}):')
    hdr = (f"  {'Algorithm':<18s} {'Lat(ms)':>10s} {'Eng(mJ)':>10s} "
           f"{'Priv':>8s} {'SLA%':>8s}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for alg in alg_names:
        s = summary[alg]
        tag = '*' if alg == 'DQN-ES' else ' '
        print(f'  {alg+tag:<18s} '
              f'{s["avg_latency_ms"]["mean"]:>10.2f} '
              f'{s["avg_energy_mj"]["mean"]:>10.4f} '
              f'{s["avg_privacy_risk"]["mean"]:>8.4f} '
              f'{s["sla_violation_pct"]["mean"]:>8.2f}')
    print(f'\n[MIT-BIH] Saved {csv_path}')
    print(f'[MIT-BIH] Saved {json_path}')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir',    type=str, default=None)
    p.add_argument('--results-dir', type=str, default=None)
    p.add_argument('--n_runs',      type=int, default=MITBIH_N_RUNS)
    p.add_argument('--workers',     type=int, default=None,
                   help='Worker processes (default: cpu_count() - 1)')
    p.add_argument('--smoke',       action='store_true',
                   help='1-run smoke test (truncates trace to verify pipeline)')
    p.add_argument('--max-tasks',   type=int, default=None,
                   help='Cap trace length (e.g., 200 for fast verification)')
    p.add_argument('--algorithms', nargs='+', default=None,
                   help='Registry names (main or experiment registry); '
                        'default = the 9 main algorithms')
    args = p.parse_args()

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    data_dir     = Path(args.data_dir)    if args.data_dir    else project_root / 'data'
    results_dir  = Path(args.results_dir) if args.results_dir else project_root / 'results'

    run_mitbih_trace(
        data_dir, results_dir,
        n_runs=args.n_runs,
        workers=args.workers,
        smoke=args.smoke,
        max_tasks=args.max_tasks,
        algorithms=args.algorithms,
    )


if __name__ == '__main__':
    # Required on Windows when using multiprocessing.spawn
    mp.freeze_support()
    main()
