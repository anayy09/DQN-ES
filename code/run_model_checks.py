"""
run_model_checks.py — independent closed-form checks of the simulator's
per-component model (replaces the former simulator-validation table).

Every expected value is computed here from the model equations and the
input parameters (hardware profiles, config).  Nothing expected is taken
from a simulator output.  The simulator's values come from
core/offload_model.offload_outcome() on a fixed one-wearable, one-edge
scenario, and from the FIFO queue in core/network.py driven by synthetic
arrivals.

Checks
  1. uplink transmission   t_tx  = D / (B log2(1 + P_rad h / N0)), P_rad the
     radiated power (18.5 dBm, datasheet Table 6-2)
  2. propagation           t_prop = d / v
  3. remote compute        t_proc = C / f_edge
  4. downlink              t_dl = d / v + S / (B log2(1 + P_dl h / N0))
  5. offload latency and wearable energy (sum of 1-4; E = P_tx t_tx +
     P_rx t_rx + P_wait (L - t_tx - t_rx), each P = datasheet current x 3.3 V:
     TX 283 mA, RX 88 mA (Table 5-7), wait 47.6 mA (Table 5-9)), and each
     energy component separately
  6. local latency and energy  L = C / f_w,  E = P_cmp C / f_w
     (P_cmp = 65.9 mA x 3.3 V, Table 5-9)
  7. queue: Poisson arrivals + exponential service through the FIFO node
     -> mean wait vs M/M/1  W_q = rho / (mu - lambda)
  8. queue: Poisson arrivals + deterministic service
     -> mean wait vs M/D/1  W_q = rho / (2 mu (1 - rho))

Usage:  python code/run_model_checks.py [--results-dir DIR]
Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from src.config import (DOWNLINK_TX_POWER_W, RESULT_SIZE_BITS,
                        WEARABLE_COMPUTE_CURRENT_A, WEARABLE_CPU_FREQ_HZ,
                        WEARABLE_RX_CURRENT_A, WEARABLE_SUPPLY_V,
                        WEARABLE_TX_CURRENT_A, WEARABLE_TX_RADIATED_DBM,
                        WEARABLE_WAIT_CURRENT_A)
from src.core.hardware_profiles import EDGE_GATEWAY_RPI4, WEARABLE_ESP32
from src.core.network import NetworkLink, NetworkNode, NetworkTopology
from src.core.offload_model import offload_outcome
from src.core.task import HealthcareTask

# Physical constants and scenario (inputs, stated here)
K_B = 1.38e-23            # Boltzmann constant used by the model, J/K
T_K = 290.0               # noise temperature, K
NOISE_FLOOR_DBM = -100.0  # receiver noise floor, dBm
V_PROP = 2e8              # propagation speed, m/s
DIST_M = 10.0             # wearable-edge distance, m
ALPHA = 3.0               # path-loss exponent
D_BITS = 4_000_000        # payload
C_CYCLES = 150_000_000    # task cycles


def shannon(p_w: float, b_hz: float, d_m: float, alpha: float) -> float:
    n0 = max(K_B * T_K * b_hz, 10 ** ((NOISE_FLOOR_DBM - 30) / 10))
    h = (1.0 / max(d_m, 1.0)) ** alpha
    return b_hz * math.log2(1 + p_w * h / n0)


def scenario():
    topo = NetworkTopology()
    topo.add_node(NetworkNode(0, 'w', 'wearable', WEARABLE_ESP32))
    topo.add_node(NetworkNode(1, 'edge', 'edge', EDGE_GATEWAY_RPI4))
    topo.add_link(NetworkLink(0, 1, DIST_M / 1000.0, ALPHA))
    task = HealthcareTask(task_id=0, device_id=0, timestamp=0.0,
                          data_size_bits=D_BITS, cpu_cycles=C_CYCLES,
                          max_delay_s=0.5, privacy_sensitivity=0.9,
                          ci_score=0.5, task_type='ecg_analysis')
    return topo, task


def check_components() -> list:
    topo, task = scenario()
    w, e = WEARABLE_ESP32, EDGE_GATEWAY_RPI4
    # Datasheet-based wearable powers (G1-3), rebuilt here from the config
    # currents: radiated power for the rate, supply draw for the energy.
    p_rad = 10 ** (WEARABLE_TX_RADIATED_DBM / 10) / 1000.0
    v = WEARABLE_SUPPLY_V
    p_tx, p_rx = WEARABLE_TX_CURRENT_A * v, WEARABLE_RX_CURRENT_A * v
    p_wait, p_cmp = WEARABLE_WAIT_CURRENT_A * v, WEARABLE_COMPUTE_CURRENT_A * v
    r_ul = shannon(p_rad, w.bandwidth_hz, DIST_M, ALPHA)
    r_dl = shannon(DOWNLINK_TX_POWER_W, w.bandwidth_hz, DIST_M, ALPHA)
    t_tx = D_BITS / r_ul
    t_prop = DIST_M / V_PROP
    t_proc = C_CYCLES / e.cpu_freq_hz
    t_rx = RESULT_SIZE_BITS / r_dl
    t_dl = t_prop + t_rx
    lat = t_tx + t_prop + t_proc + t_dl        # empty queue
    eng = p_tx * t_tx + p_rx * t_rx + p_wait * (lat - t_tx - t_rx)
    lat_local = C_CYCLES / WEARABLE_CPU_FREQ_HZ
    eng_local = p_cmp * C_CYCLES / WEARABLE_CPU_FREQ_HZ

    off = offload_outcome(topo, task, 1)
    loc = offload_outcome(topo, task, 0)
    return [
        ('uplink t_tx (ms)', t_tx * 1e3, off.t_tx * 1e3),
        ('propagation t_prop (ms)', t_prop * 1e3, off.t_prop * 1e3),
        ('edge compute t_proc (ms)', t_proc * 1e3, off.t_proc * 1e3),
        ('downlink t_dl (ms)', t_dl * 1e3, off.t_dl * 1e3),
        ('offload latency (ms)', lat * 1e3, off.latency_s * 1e3),
        ('offload wearable energy (mJ)', eng * 1e3, off.energy_j * 1e3),
        ('local latency (ms)', lat_local * 1e3, loc.latency_s * 1e3),
        ('local energy (mJ)', eng_local * 1e3, loc.energy_j * 1e3),
        ('offload TX energy (mJ)', p_tx * t_tx * 1e3, off.e_tx * 1e3),
        ('offload wait energy (mJ)', p_wait * (lat - t_tx - t_rx) * 1e3, off.e_idle * 1e3),
        ('offload RX energy (mJ)', p_rx * t_rx * 1e3, off.e_rx * 1e3),
    ]


def simulate_fifo(lam: float, service: np.ndarray, rng) -> float:
    """Mean FIFO wait of Poisson(lam) arrivals through topology.reserve()."""
    topo = NetworkTopology()
    topo.add_node(NetworkNode(1, 'edge', 'edge', EDGE_GATEWAY_RPI4))
    arrivals = np.cumsum(rng.exponential(1.0 / lam, size=len(service)))
    waits = np.empty(len(service))
    for i, (a, s) in enumerate(zip(arrivals, service)):
        topo.advance_time(float(a))
        waits[i], _ = topo.reserve(1, float(a), float(s))
    warm = len(service) // 10              # drop the initial transient
    return float(waits[warm:].mean())


def check_queues(n: int = 200_000, seed: int = 7) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for rho in (0.3, 0.6, 0.85):
        mu = 10.0
        lam = rho * mu
        wq_mm1 = rho / (mu - lam)
        sim = simulate_fifo(lam, rng.exponential(1.0 / mu, size=n), rng)
        out.append((f'M/M/1 mean wait rho={rho} (ms)', wq_mm1 * 1e3, sim * 1e3))
        wq_md1 = rho / (2 * mu * (1 - rho))
        sim = simulate_fifo(lam, np.full(n, 1.0 / mu), rng)
        out.append((f'M/D/1 mean wait rho={rho} (ms)', wq_md1 * 1e3, sim * 1e3))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--results-dir', type=str, default=None)
    args = p.parse_args()
    out_dir = (Path(args.results_dir) if args.results_dir
               else Path(BASE).parent / 'results')
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, ok = [], True
    # Deterministic components: exact agreement expected (rel. 1e-9).
    # Queues: sampling error of a 200k-arrival run; tolerance 5 %.
    for group, tol in ((check_components(), 1e-9), (check_queues(), 0.05)):
        for name, expected, simulated in group:
            rel = abs(simulated - expected) / max(abs(expected), 1e-15)
            passed = rel <= tol
            ok &= passed
            rows.append({'check': name, 'expected': expected,
                         'simulated': simulated, 'rel_error': rel,
                         'tolerance': tol, 'pass': passed})

    print(f"{'check':<34} {'expected':>12} {'simulated':>12} {'rel.err':>10}")
    for r in rows:
        print(f"{r['check']:<34} {r['expected']:>12.5f} {r['simulated']:>12.5f} "
              f"{r['rel_error']:>10.2e} {'ok' if r['pass'] else 'FAIL'}")
    (out_dir / 'model_checks.json').write_text(json.dumps(rows, indent=1),
                                               encoding='utf-8')
    print(f"\n{'all checks passed' if ok else 'CHECK FAILED'}; "
          f"wrote {out_dir / 'model_checks.json'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
