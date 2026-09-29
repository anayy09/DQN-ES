"""
scheduling_overhead.py — decision cost vs fog-node count (plan E4, E13).

For each arm in ROBUSTNESS_ARMS and each M in SCALING_FOG_COUNTS, runs
SCALING_TIMING_RUNS replicates at N tasks and reports the per-decision
wall-clock split recorded by the schedulers (last_decision_info):

  bounds    estimate_feasible_bounds (normalisation sweep over local + M+2)
  forward   DQN forward pass on the decision path (0 on exploratory steps)
  enum      enumeration of the candidate set (K for DQN-ES / Random-K,
            M+2 for ES-only)
  update    replay push + minibatch update (DQN arms; off the decision path)
  select    the whole select_node() call as timed by the environment

as median / p95 / p99 in ms over all decisions, plus the analytic parameter
count, FLOPs and memory of the DQN at that M (machine_info.dqn_cost) and the
host CPU.  Timing runs are serial by default (workers=1) so that concurrent
workers do not inflate the timings.  There is no embedded-hardware
measurement (ledger D3): numbers are for the named laptop CPU.

Outputs (results_dir):
  scheduling_overhead_summary.csv   one row per (arm, M, component)
  scheduling_overhead.json          same + host info + DQN cost per M
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

import numpy as np

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from src.analysis.machine_info import dqn_cost, host_info
from src.config import (
    DQN_BATCH_SIZE,
    DQN_HIDDEN_DIM,
    DQN_REPLAY_CAPACITY,
    PRIMARY_SCALE,
    ROBUSTNESS_ARMS,
    SCALING_FOG_COUNTS,
    SCALING_TIMING_RUNS,
)
from src.simulation.replicate import build_synthetic_replicate, run_scheduler

COMPONENTS = ['t_bounds_ms', 't_forward_ms', 't_enum_ms', 't_update_ms',
              'scheduling_overhead_ms']
DQN_ARMS_PREFIX = ('DQN-ES', 'PSO+DQN', 'DQN-only')


def _run_cell(payload: tuple) -> tuple:
    """payload = (arm, run_id, n_tasks, n_fog) -> (arm, n_fog, run_id, times, metrics)."""
    arm, run_id, n_tasks, n_fog = payload
    from src.config import get_registry
    seeds, topo, tasks = build_synthetic_replicate(run_id, n_tasks, 'mixed',
                                                   n_fog_nodes=n_fog)
    res, _ = run_scheduler(get_registry('all')[arm], topo, tasks, seeds)
    times = {c: [r[c] for r in res if r.get(c) is not None] for c in COMPONENTS}
    metrics = {
        'avg_latency_ms': float(np.mean([r['latency_ms'] for r in res])),
        'avg_energy_mj': float(np.mean([r['energy_mj'] for r in res])),
        'avg_privacy_risk': float(np.mean([r['privacy_risk'] for r in res])),
        'sla_violation_pct': 100.0 * float(np.mean([r['sla_violated'] for r in res])),
    }
    return arm, n_fog, run_id, times, metrics


def _pct(x):
    a = np.asarray(x, dtype=float)
    if a.size == 0:
        return {'median': None, 'p95': None, 'p99': None, 'n': 0}
    return {'median': float(np.median(a)), 'p95': float(np.percentile(a, 95)),
            'p99': float(np.percentile(a, 99)), 'n': int(a.size)}


def run_overhead_analysis(
    results_dir: Path,
    n_runs: int = SCALING_TIMING_RUNS,
    n_tasks: int = PRIMARY_SCALE,
    workers: int | None = 1,
    fog_counts=SCALING_FOG_COUNTS,
    arms=ROBUSTNESS_ARMS,
) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    payloads = [(a, r, n_tasks, m) for m in fog_counts for a in arms
                for r in range(n_runs)]
    workers = max(1, min(workers or 1, len(payloads)))
    host = host_info()
    print(f'[OVERHEAD] {len(payloads)} cells, workers={workers}, '
          f'CPU: {host["cpu"]}')

    times = defaultdict(lambda: defaultdict(list))   # (arm, M) -> comp -> list
    beh = defaultdict(list)
    t0 = time.time()
    if workers == 1:
        results = map(_run_cell, payloads)
    else:
        pool = mp.get_context('spawn').Pool(processes=workers)
        results = pool.imap_unordered(_run_cell, payloads, chunksize=1)
    for arm, m, rid, tt, met in results:
        for c, v in tt.items():
            times[(arm, m)][c].extend(v)
        beh[(arm, m)].append(met)
    if workers > 1:
        pool.close()
        pool.join()
    print(f'[OVERHEAD] done in {time.time() - t0:.1f}s')

    rows, out = [], {'host': host, 'n_runs': n_runs, 'n_tasks': n_tasks,
                     'workers': workers, 'cells': [], 'dqn_cost': {}}
    for m in fog_counts:
        n_actions = m + 2
        state_dim = 2 + 4 * n_actions
        out['dqn_cost'][str(m)] = dqn_cost(state_dim, n_actions, DQN_HIDDEN_DIM,
                                           DQN_BATCH_SIZE, DQN_REPLAY_CAPACITY)
        for arm in arms:
            comp = {c: _pct(times[(arm, m)][c]) for c in COMPONENTS}
            b = beh[(arm, m)]
            bmean = {k: float(np.mean([x[k] for x in b])) for k in b[0]} if b else {}
            cell = {'arm': arm, 'M': m, 'components': comp, 'behaviour_mean': bmean}
            if arm.startswith(DQN_ARMS_PREFIX):
                cell['dqn_cost'] = out['dqn_cost'][str(m)]
            out['cells'].append(cell)
            for c, s in comp.items():
                rows.append({'arm': arm, 'M': m, 'component': c.replace('_ms', ''),
                             'n_decisions': s['n'], 'median_ms': s['median'],
                             'p95_ms': s['p95'], 'p99_ms': s['p99'],
                             'cpu': host['cpu']})

    csv_path = results_dir / 'scheduling_overhead_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    json_path = results_dir / 'scheduling_overhead.json'
    json_path.write_text(json.dumps(out, indent=1), encoding='utf-8')

    print(f'\n[OVERHEAD] CPU: {host["cpu"]}  (median / p95 / p99 ms per decision)')
    for cell in out['cells']:
        c = cell['components']
        f = lambda k: (f"{c[k]['median']:.3f}/{c[k]['p95']:.3f}/{c[k]['p99']:.3f}"
                       if c[k]['n'] else '-')
        print(f"  M={cell['M']:<3d} {cell['arm']:<15s} bounds {f('t_bounds_ms')}  "
              f"fwd {f('t_forward_ms')}  enum {f('t_enum_ms')}  "
              f"upd {f('t_update_ms')}  select {f('scheduling_overhead_ms')}")
    for m, d in out['dqn_cost'].items():
        print(f"  DQN at M={m}: params={d['params']}  forward FLOPs={d['forward_flops']}  "
              f"update FLOPs~{d['update_flops_approx']}  memory~{d['memory_bytes_approx']/1e6:.1f} MB")
    print(f'[OVERHEAD] Saved {csv_path} and {json_path}')
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', type=str, default=None)
    p.add_argument('--n_runs', type=int, default=SCALING_TIMING_RUNS)
    p.add_argument('--n_tasks', type=int, default=PRIMARY_SCALE)
    p.add_argument('--workers', type=int, default=1,
                   help='1 (default) keeps timings free of worker contention')
    p.add_argument('--fog-counts', type=int, nargs='+', default=SCALING_FOG_COUNTS)
    p.add_argument('--arms', nargs='+', default=ROBUSTNESS_ARMS)
    args = p.parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    results_dir = (Path(args.results_dir) if args.results_dir
                   else project_root / 'results')
    run_overhead_analysis(results_dir, args.n_runs, args.n_tasks, args.workers,
                          args.fog_counts, args.arms)


if __name__ == '__main__':
    mp.freeze_support()
    main()
