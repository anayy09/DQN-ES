"""
run_mitbih_psodqn.py — run MIT-BIH real-trace evaluation for PSO+DQN only,
then merge the result into the existing results/mitbih_trace_raw.json.

Runs MITBIH_N_RUNS (=30) replicates in parallel and adds the PSO+DQN entry
without touching any of the 8 existing algorithm entries.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PROJECT_ROOT = Path(BASE).parent
RESULTS_DIR  = PROJECT_ROOT / 'results'
DATA_DIR     = PROJECT_ROOT / 'data'

from src.config import (
    GLOBAL_SEED,
    MITBIH_DEADLINE_S,
    MITBIH_N_RUNS,
    MITBIH_PAYLOAD_BITS,
    MITBIH_RHO,
    N_FOG_NODES,
    N_WEARABLES,
    get_full_algorithm_registry,
    replicate_seeds,
    seed_global_rngs,
)
from src.core.task import TASK_PROFILES, HealthcareTask
from src.data_ingestion.parse_mitbih import load_mitbih_events
from src.simulation.environment import OffloadingEnvironment
from src.simulation.replicate import build_topology, run_scheduler

# -------------------------------------------------------------------------
# Worker-side globals
# -------------------------------------------------------------------------
_WORKER_EVENTS: list | None = None


def _worker_init(events_list: list) -> None:
    global _WORKER_EVENTS
    _WORKER_EVENTS = events_list


def _events_to_tasks(events: list, topo, rng) -> list:
    import random as _r
    wids = [nid for nid, n in topo.nodes.items() if n.node_type == 'wearable']
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


def _run_cell(payload: tuple) -> tuple:
    import random as _r
    alg_name, run_id = payload
    t0 = time.time()

    seeds = replicate_seeds(run_id, 0)
    seed_global_rngs(seeds['base'])
    local_rng = _r.Random(seeds['task'])

    topo = build_topology(run_id, 0)
    tasks = _events_to_tasks(_WORKER_EVENTS, topo, local_rng)

    registry = get_full_algorithm_registry()
    res, _ = run_scheduler(registry[alg_name], topo, tasks, seeds)

    if not res:
        return alg_name, run_id, None, time.time() - t0

    metrics = {
        'avg_latency_ms':    mean(r['latency_ms']    for r in res),
        'avg_energy_mj':     mean(r['energy_mj']     for r in res),
        'avg_privacy_risk':  mean(r['privacy_risk']  for r in res),
        'sla_violation_pct': 100.0 * sum(r['sla_violated']
                                         for r in res) / len(res),
    }
    return alg_name, run_id, metrics, time.time() - t0


def main():
    ALG = 'PSO+DQN'
    n_runs  = MITBIH_N_RUNS   # 30
    workers = max(1, (os.cpu_count() or 2) - 1)

    json_path = RESULTS_DIR / 'mitbih_trace_raw.json'
    if not json_path.exists():
        raise FileNotFoundError(f'mitbih_trace_raw.json not found at {json_path}. '
                                'Run the full MIT-BIH pipeline first.')

    print(f'[MITBIH-PSO+DQN] Loading events from {DATA_DIR} ...')
    events = load_mitbih_events(str(DATA_DIR))
    print(f'[MITBIH-PSO+DQN] Loaded {len(events)} window events.')

    payloads = [(ALG, rid) for rid in range(n_runs)]
    print(f'[MITBIH-PSO+DQN] Dispatching {len(payloads)} cells over {workers} workers.')

    raw: list = []
    t_start = time.time()
    completed = 0

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=workers, initializer=_worker_init,
                  initargs=(events,)) as pool:
        for alg, rid, metrics, dt in pool.imap_unordered(
                _run_cell, payloads, chunksize=1):
            completed += 1
            if metrics is not None:
                raw.append(metrics)
            wall = time.time() - t_start
            eta  = wall / completed * (n_runs - completed) if completed else 0
            print(f'  [{completed:2d}/{n_runs}] run={rid:<3d}  '
                  f'dt={dt:6.1f}s  wall={wall/60:5.1f}m  eta={eta/60:5.1f}m',
                  flush=True)

    elapsed = time.time() - t_start
    print(f'\n[MITBIH-PSO+DQN] Finished {n_runs} runs in {elapsed/60:.1f} min.')

    # Aggregate
    agg = {}
    for k in ['avg_latency_ms', 'avg_energy_mj', 'avg_privacy_risk', 'sla_violation_pct']:
        vs = np.array([r[k] for r in raw], dtype=float)
        agg[k] = {'mean': float(vs.mean()), 'std': float(vs.std()),
                  'samples': vs.tolist()}
        print(f'  {k}: {vs.mean():.4f} ± {vs.std():.4f}')

    # Merge into existing JSON
    with open(json_path, 'r', encoding='utf-8') as fh:
        existing = json.load(fh)

    existing[ALG] = agg

    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(existing, fh, indent=2)
    print(f'\n[MITBIH-PSO+DQN] Merged {ALG} into {json_path}')
    print('[MITBIH-PSO+DQN] Done.')


if __name__ == '__main__':
    mp.freeze_support()
    main()
