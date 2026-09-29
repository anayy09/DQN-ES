"""
Per-task raw logs and episode-level extras (plan E5, E8, E11; self-review F7).

write_raw_log()          one gzip CSV per (experiment, N, algorithm, run) with
                         every observable a passive adversary could see
                         (device, timestamp, destination, payload and result
                         size, response time), the CI label, the latency and
                         energy components, and the DQN decision diagnostics.
steady_state_privacy()   mean R_P over tasks after each device's first W
                         decisions (W = 50, the entropy window), so the
                         empty-window warm-up does not bias R_P.

The summary metrics (avg latency / energy / R_P / SLA %) are unchanged.
"""

from __future__ import annotations

import csv
import gzip
import math
import os
import re
from pathlib import Path
from typing import List, Optional

from src.algorithms.base_scheduler import WINDOW_SIZE

RAW_LOG_COLUMNS = [
    # replicate
    'algorithm', 'run_id', 'n_tasks',
    # task (what the adversary observes + labels)
    'task_id', 'device_id', 'device_decision_index', 'timestamp',
    'task_type', 'ci_score', 'ci_tier', 'ci_visible', 'payload_bits',
    'result_bits',
    'sla_deadline_ms', 'attack_prob',
    # decision
    'assigned_node', 'node_type', 'explored', 'epsilon', 'q_argmax_node',
    'exec_q_rank',
    # outcome
    'latency_ms', 'completion_time_s', 'latency_tx_ms', 'latency_prop_ms',
    'latency_queue_ms', 'latency_compute_ms', 'latency_downlink_ms',
    'energy_mj', 'energy_tx_mj', 'energy_idle_mj', 'energy_rx_mj',
    'energy_compute_mj', 'privacy_risk', 'sla_violated',
    'scheduling_overhead_ms', 't_bounds_ms', 't_forward_ms', 't_enum_ms',
    't_update_ms',
]


def steady_state_privacy(results: List[dict],
                         window: int = WINDOW_SIZE) -> Optional[float]:
    """Mean privacy_risk over tasks with device_decision_index >= window."""
    vals = [r['privacy_risk'] for r in results
            if r.get('device_decision_index', -1) >= window]
    return (sum(vals) / len(vals)) if vals else None


def add_steady_state(metrics: dict, results: List[dict],
                     window: int = WINDOW_SIZE) -> dict:
    """Add avg_privacy_risk_ss / n_ss_tasks when any task is past warm-up."""
    ss = steady_state_privacy(results, window)
    n_ss = sum(1 for r in results
               if r.get('device_decision_index', -1) >= window)
    if ss is not None:
        metrics['avg_privacy_risk_ss'] = ss
    metrics['n_ss_tasks'] = n_ss
    return metrics


def add_queue_metrics(metrics: dict, results: List[dict]) -> dict:
    """
    avg_queue_ms: mean FIFO waiting time per task (D16).
    edge_utilisation: service time on edge nodes / arrival span (achieved
    utilisation; E7's rho target is the offered load).
    """
    if not results:
        return metrics
    metrics['avg_queue_ms'] = sum(r['latency_queue_ms'] for r in results) / len(results)
    ts = [r['timestamp'] for r in results]
    span = max(ts) - min(ts)
    busy = sum(r['latency_compute_ms'] for r in results
               if r.get('node_type') == 'edge') / 1000.0
    metrics['edge_utilisation'] = busy / span if span > 0 else 0.0
    return metrics


def _safe(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')


def raw_log_path(raw_root: Path, n_tasks: int, algorithm: str,
                 run_id: int) -> Path:
    # Flat per-scale layout keeps paths short (Windows MAX_PATH).
    return (Path(raw_root) / f'n{n_tasks}'
            / f'{_safe(algorithm)}__run{run_id:02d}.csv.gz')


def _long_path(path: Path) -> Path:
    """Use the extended-length prefix on Windows past MAX_PATH (260 chars)."""
    prefix = '\\\\?\\'
    if os.name == 'nt':
        full = str(path.resolve())
        if len(full) >= 240 and not full.startswith(prefix):
            return Path(prefix + full)
    return path


def _fmt(v):
    if v is None:
        return ''
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return '' if math.isnan(v) else f'{v:.6g}'
    return v


def write_raw_log(results: List[dict], path: Path, algorithm: str,
                  run_id: int, n_tasks: int) -> Path:
    path = _long_path(Path(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {'algorithm': algorithm, 'run_id': run_id, 'n_tasks': n_tasks}
    with gzip.open(path, 'wt', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh)
        wr.writerow(RAW_LOG_COLUMNS)
        for r in results:
            row = {**r, **meta}
            wr.writerow([_fmt(row.get(c)) for c in RAW_LOG_COLUMNS])
    return path
