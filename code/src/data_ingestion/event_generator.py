"""
event_generator.py
------------------
Combine all data sources into a unified simulation event stream.

Each task is modelled as the 4-tuple (D_i, C_i, T_max_i, rho_i) required
by the Bio-Inspired Adaptive Task Offloading system, plus the Criticality
Index Phi_i and optional attack_probability from network security datasets.

Public API
----------
generate_event_stream(n_tasks, n_devices, mendeley_events=None,
                      mitbih_events=None, ciciot_events=None, seed=42)
    -> List[SimulationTask]

generate_synthetic_tasks(n, ci_distribution='mixed', seed=42)
    -> List[SimulationTask]
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict
from typing import List, Optional


# ---------------------------------------------------------------------------
# Task-type parameter profiles: single source in src/core/task.py
# (ECG payload from src.config.ECG_PAYLOAD_BITS).
# ---------------------------------------------------------------------------
from src.core.task import TASK_PROFILES  # noqa: E402

# Task-type proportions for synthetic generation
TASK_TYPE_WEIGHTS = {
    "ecg_analysis":    0.30,
    "spo2_monitoring": 0.40,
    "bp_analysis":     0.20,
    "multi_vital":     0.10,
}


# ---------------------------------------------------------------------------
# SimulationTask dataclass
# ---------------------------------------------------------------------------
@dataclass
class SimulationTask:
    task_id:             int
    device_id:           int
    timestamp:           float
    ci_score:            float          # Phi_i ∈ [0,1]
    data_size_bits:      int            # D_i
    cpu_cycles:          int            # C_i
    max_delay_s:         float          # T_max_i
    privacy_sensitivity: float          # rho_i
    source:              str            # 'mendeley' | 'mitbih' | 'synthetic'
    attack_probability:  float = 0.0   # from CICIoMT / MedSec
    task_type:           str   = "ecg_analysis"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _poisson_timestamps(n: int, lam: float, rng: random.Random) -> list:
    """
    Generate n arrival timestamps via a Poisson process with rate lam.
    Inter-arrival times are Exponential(lam).
    """
    timestamps = []
    t = 0.0
    for _ in range(n):
        t += rng.expovariate(lam)
        timestamps.append(round(t, 6))
    return timestamps


def _sample_attack_prob(ciciot_events: Optional[list],
                        rng: random.Random) -> float:
    """Return a random attack_probability from the CICIoT pool, or 0.0."""
    if not ciciot_events:
        return 0.0
    event = rng.choice(ciciot_events)
    return float(event.get("attack_probability", 0.0))


def _ci_for_distribution(tier: str, rng: random.Random) -> float:
    """Sample a CI score from the specified CI tier."""
    if tier == "all_high":
        # Fix B: ICU scenario — all tasks arrive with CI uniformly in [0.8, 1.0]
        return round(rng.uniform(0.8, 1.0), 4)
    elif tier == "high":
        return round(rng.uniform(0.7, 1.0), 4)
    elif tier == "medium":
        return round(rng.uniform(0.3, 0.7), 4)
    else:  # low
        return round(rng.uniform(0.0, 0.3), 4)


def _task_from_mendeley(event: dict,
                        task_id: int,
                        device_id: int,
                        timestamp: float,
                        attack_prob: float) -> SimulationTask:
    """Convert a Mendeley event dict into a SimulationTask."""
    return SimulationTask(
        task_id             = task_id,
        device_id           = device_id,
        timestamp           = timestamp,
        ci_score            = float(event.get("ci_score", 0.5)),
        data_size_bits      = int(event.get("data_size_bits", 8_000_000)),
        cpu_cycles          = int(event.get("cpu_cycles",     8_000_000)),
        max_delay_s         = float(event.get("max_delay_s",  0.8)),
        privacy_sensitivity = float(event.get("privacy_sensitivity", 0.85)),
        source              = "mendeley",
        attack_probability  = attack_prob,
        task_type           = str(event.get("task_type", "multi_vital")),
    )


def _task_from_mitbih(event: dict,
                      task_id: int,
                      device_id: int,
                      timestamp: float,
                      attack_prob: float) -> SimulationTask:
    """Convert a MIT-BIH window event dict into a SimulationTask."""
    ci = float(event.get("ci_score", 0.5))
    # Scale CPU cycles from the event; data payload fixed for ECG.
    # Default updated to 150 M cycles to match realistic CNN inference on
    # a Cortex-M / ESP32-class wearable (see TASK_PROFILES['ecg_analysis']).
    profile = TASK_PROFILES["ecg_analysis"]
    cpu = int(event.get("cpu_cycles_ecg_inference", profile["cpu_cycles"]))
    return SimulationTask(
        task_id             = task_id,
        device_id           = device_id,
        timestamp           = timestamp,
        ci_score            = ci,
        data_size_bits      = profile["data_size_bits"],
        cpu_cycles          = cpu,
        max_delay_s         = profile["max_delay_s"],
        privacy_sensitivity = profile["privacy_sensitivity"],
        source              = "mitbih",
        attack_probability  = attack_prob,
        task_type           = "ecg_analysis",
    )


def _make_synthetic_task(task_id: int,
                         device_id: int,
                         timestamp: float,
                         ci: float,
                         task_type: str,
                         attack_prob: float) -> SimulationTask:
    """Build a synthetic SimulationTask from a CI score and task type."""
    profile = TASK_PROFILES[task_type]
    return SimulationTask(
        task_id             = task_id,
        device_id           = device_id,
        timestamp           = timestamp,
        ci_score            = ci,
        data_size_bits      = profile["data_size_bits"],
        cpu_cycles          = profile["cpu_cycles"],
        max_delay_s         = profile["max_delay_s"],
        privacy_sensitivity = profile["privacy_sensitivity"],
        source              = "synthetic",
        attack_probability  = attack_prob,
        task_type           = task_type,
    )


# ---------------------------------------------------------------------------
# generate_synthetic_tasks
# ---------------------------------------------------------------------------
def generate_synthetic_tasks(
        n: int,
        ci_distribution: str = "mixed",
        seed: int = 42,
) -> List[SimulationTask]:
    """
    Generate n purely synthetic SimulationTask objects.

    Parameters
    ----------
    n : int
        Number of tasks to generate.
    ci_distribution : str
        'mixed'  → 20% high CI (>0.7), 60% medium, 20% low
        'high'   → all high CI
        'medium' → all medium CI
        'low'    → all low CI
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[SimulationTask]
    """
    rng = random.Random(seed)

    # CI tier distribution
    if ci_distribution == "mixed":
        tiers = rng.choices(
            ["high", "medium", "low"],
            weights=[0.20, 0.60, 0.20],
            k=n,
        )
    elif ci_distribution == "all_high":
        # Fix B: ICU scenario — all tasks have CI in [0.8, 1.0]
        tiers = ["all_high"] * n
    else:
        tiers = [ci_distribution] * n

    # Task types
    task_types = rng.choices(
        list(TASK_TYPE_WEIGHTS.keys()),
        weights=list(TASK_TYPE_WEIGHTS.values()),
        k=n,
    )

    # Poisson timestamps
    lam = n / 300.0
    timestamps = _poisson_timestamps(n, lam, rng)

    tasks = []
    for i in range(n):
        ci   = _ci_for_distribution(tiers[i], rng)
        t    = timestamps[i]
        tt   = task_types[i]
        dev  = rng.randint(0, max(1, n // 10) - 1)
        tasks.append(
            _make_synthetic_task(
                task_id      = i,
                device_id    = dev,
                timestamp    = t,
                ci           = ci,
                task_type    = tt,
                attack_prob  = 0.0,
            )
        )

    print(f"[EventGen] Generated {len(tasks)} synthetic tasks "
          f"(ci_distribution='{ci_distribution}').")
    return tasks


# ---------------------------------------------------------------------------
# generate_event_stream
# ---------------------------------------------------------------------------
def generate_event_stream(
        n_tasks:        int,
        n_devices:      int,
        mendeley_events: Optional[list] = None,
        mitbih_events:   Optional[list] = None,
        ciciot_events:   Optional[list] = None,
        seed:           int = 42,
) -> List[SimulationTask]:
    """
    Blend real events from Mendeley / MIT-BIH with synthetic tasks to reach
    n_tasks, assign Poisson arrival timestamps, and attach attack
    probabilities sampled from CICIoMT.

    Parameters
    ----------
    n_tasks : int
        Total number of tasks in the output stream.
    n_devices : int
        Number of IoT devices.  Device IDs are drawn from [0, n_devices).
    mendeley_events : list | None
        Output of parse_mendeley.load_mendeley_events().
    mitbih_events : list | None
        Output of parse_mitbih.load_mitbih_events().
    ciciot_events : list | None
        Output of parse_ciciot.load_ciciot_events(); used only for
        attack_probability sampling.
    seed : int
        Random seed.

    Returns
    -------
    List[SimulationTask] sorted by timestamp.
    """
    rng = random.Random(seed)

    # --- Poisson arrival timestamps for the full stream ---
    lam = n_tasks / 300.0  # overall rate: n_tasks spread over ~5 minutes
    timestamps = _poisson_timestamps(n_tasks, lam, rng)
    rng.shuffle(timestamps)   # decouple timestamp ordering from source order

    tasks: List[SimulationTask] = []
    task_id = 0

    # --- Pool real events ---
    mendeley_pool = list(mendeley_events) if mendeley_events else []
    mitbih_pool   = list(mitbih_events)   if mitbih_events   else []

    rng.shuffle(mendeley_pool)
    rng.shuffle(mitbih_pool)

    total_real = len(mendeley_pool) + len(mitbih_pool)
    n_real     = min(n_tasks, total_real)
    n_synthetic = n_tasks - n_real

    print(f"\n[EventGen] Target tasks    : {n_tasks}")
    print(f"[EventGen] Real (Mendeley) : {len(mendeley_pool)} available")
    print(f"[EventGen] Real (MIT-BIH)  : {len(mitbih_pool)} available")
    print(f"[EventGen] Will use real   : {n_real}")
    print(f"[EventGen] Will synthesise : {n_synthetic}")

    # --- Decide split between Mendeley and MIT-BIH ---
    if mendeley_pool and mitbih_pool:
        n_mendeley = int(n_real * len(mendeley_pool) /
                         (len(mendeley_pool) + len(mitbih_pool)))
        n_mitbih   = n_real - n_mendeley
    elif mendeley_pool:
        n_mendeley = n_real
        n_mitbih   = 0
    else:
        n_mendeley = 0
        n_mitbih   = n_real

    # --- Mendeley tasks ---
    for i in range(n_mendeley):
        ev  = mendeley_pool[i % len(mendeley_pool)]
        ts  = timestamps[task_id]
        dev = rng.randint(0, max(1, n_devices) - 1)
        ap  = _sample_attack_prob(ciciot_events, rng)
        tasks.append(_task_from_mendeley(ev, task_id, dev, ts, ap))
        task_id += 1

    # --- MIT-BIH tasks ---
    for i in range(n_mitbih):
        ev  = mitbih_pool[i % len(mitbih_pool)]
        ts  = timestamps[task_id]
        dev = rng.randint(0, max(1, n_devices) - 1)
        ap  = _sample_attack_prob(ciciot_events, rng)
        tasks.append(_task_from_mitbih(ev, task_id, dev, ts, ap))
        task_id += 1

    # --- Synthetic tasks to reach n_tasks ---
    if n_synthetic > 0:
        # CI tier distribution: 20% high, 60% medium, 20% low
        tiers = rng.choices(
            ["high", "medium", "low"],
            weights=[0.20, 0.60, 0.20],
            k=n_synthetic,
        )
        task_types = rng.choices(
            list(TASK_TYPE_WEIGHTS.keys()),
            weights=list(TASK_TYPE_WEIGHTS.values()),
            k=n_synthetic,
        )
        for i in range(n_synthetic):
            ts  = timestamps[task_id]
            dev = rng.randint(0, max(1, n_devices) - 1)
            ci  = _ci_for_distribution(tiers[i], rng)
            ap  = _sample_attack_prob(ciciot_events, rng)
            tasks.append(
                _make_synthetic_task(
                    task_id     = task_id,
                    device_id   = dev,
                    timestamp   = ts,
                    ci          = ci,
                    task_type   = task_types[i],
                    attack_prob = ap,
                )
            )
            task_id += 1

    # --- Sort by timestamp ---
    tasks.sort(key=lambda t: t.timestamp)

    print(f"[EventGen] Final stream    : {len(tasks)} tasks "
          f"(span {tasks[-1].timestamp:.1f}s if non-empty)")
    return tasks


# ---------------------------------------------------------------------------
# CLI entry-point / smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    results_dir  = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Event Generator — smoke test")
    print("=" * 60)

    # Smoke-test with purely synthetic tasks
    tasks = generate_event_stream(
        n_tasks   = 1_000,
        n_devices = 20,
        seed      = 42,
    )

    out_path = results_dir / "event_stream_sample.json"
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump([t.to_dict() for t in tasks], fh, indent=2)
        print(f"[EventGen] Saved {len(tasks)} tasks to {out_path}")
    except Exception as exc:
        print(f"[EventGen] Failed to save JSON: {exc}")
        sys.exit(1)

    print("\n[EventGen] First 3 tasks:")
    for t in tasks[:3]:
        print(" ", t)

    # Also test generate_synthetic_tasks
    synth = generate_synthetic_tasks(n=500, ci_distribution="mixed", seed=0)
    print(f"\n[EventGen] generate_synthetic_tasks produced {len(synth)} tasks.")
    ci_vals = [t.ci_score for t in synth]
    print(f"  CI min={min(ci_vals):.3f}  max={max(ci_vals):.3f}  "
          f"mean={sum(ci_vals)/len(ci_vals):.3f}")
