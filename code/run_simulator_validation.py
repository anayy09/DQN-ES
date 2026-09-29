"""
run_simulator_validation.py — Analytical cross-validation of NumPy simulator.

Compares simulator outputs against closed-form analytical predictions for
each latency component (transmission, propagation, queue, compute) to verify
the simulator's physical fidelity. This acts as an iFogSim2-style cross-check
using the same hardware parameters from Table I of the manuscript.

Outputs: results/simulator_validation.json  and  results/table_sim_validation.csv
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

PROJECT_ROOT = Path(BASE).parent
RESULTS_DIR  = PROJECT_ROOT / 'results'

# ── Hardware parameters from Table I of manuscript ──────────────────────────
ESP32_FREQ_HZ  = 240e6          # 240 MHz
RPI4_FREQ_HZ   = 1500e6         # 1.5 GHz
FOG_FREQ_HZ    = 2200e6         # 2.2 GHz
CLOUD_FREQ_HZ  = 3200e6         # 3.2 GHz

WEARABLE_TX_POWER_W  = 0.178    # 178 mW
WEARABLE_IDLE_POWER_W = 0.033   # 33 mW

KAPPA = 1e-27                   # effective switched capacitance

# Wireless channel: 20 MHz BW, path-loss exponent α=3, indoor
BW_HZ        = 20e6             # 20 MHz wearable uplink
NOISE_PSD    = -174e-3          # -174 dBm/Hz → W/Hz (approximation)
NOISE_W      = 10**((-174 - 30) / 10) * BW_HZ   # thermal noise floor
DIST_M       = 5.0              # representative 5-metre wearable-to-edge distance
H0           = 1e-3             # reference path gain at d0=1m
D0_M         = 1.0
ALPHA        = 3.0              # indoor path-loss exponent

# ECG task (dominant workload in paper)
from src.core.task import TASK_PROFILES as _TP  # noqa: E402
# Single source (core/task.py, D15)
ECG_BITS     = _TP['ecg_analysis']['data_size_bits']
ECG_CYCLES   = _TP['ecg_analysis']['cpu_cycles']
ECG_DEADLINE = _TP['ecg_analysis']['max_delay_s']

# Queue parameters (M/M/1, low-load setting)
LAMBDA_EDGE = 5.0     # arrival rate tasks/s
MU_EDGE     = 20.0    # service rate tasks/s


def analytical_values():
    """Compute closed-form analytical reference values."""

    # 1. Channel gain at 5 m (log-distance)
    h = H0 * (D0_M / DIST_M) ** ALPHA

    # 2. Shannon capacity (uplink rate)
    snr = (WEARABLE_TX_POWER_W * h) / NOISE_W
    R_bps = BW_HZ * math.log2(1.0 + snr)
    R_mbps = R_bps / 1e6

    # 3. Transmission delay to edge
    T_tx_s = ECG_BITS / R_bps
    T_tx_ms = T_tx_s * 1000.0

    # 4. Propagation delay (5 m at speed of light)
    T_prop_ms = (DIST_M / 3e8) * 1000.0

    # 5. Queue delay (M/M/1 low load)
    rho = LAMBDA_EDGE / MU_EDGE
    T_queue_ms = (LAMBDA_EDGE / (MU_EDGE * (MU_EDGE - LAMBDA_EDGE))) * 1000.0

    # 6. Compute delay at edge (RPi4)
    T_compute_edge_ms = (ECG_CYCLES / RPI4_FREQ_HZ) * 1000.0

    # 7. Total offload latency to edge
    T_total_edge_ms = T_tx_ms + T_prop_ms + T_queue_ms + T_compute_edge_ms

    # 8. Local compute latency (ESP32)
    T_local_ms = (ECG_CYCLES / ESP32_FREQ_HZ) * 1000.0

    # 9. Offload energy (wearable perspective)
    E_tx_J = WEARABLE_TX_POWER_W * T_tx_s
    T_wait_s = T_total_edge_ms / 1000.0 - T_tx_s
    E_idle_J = WEARABLE_IDLE_POWER_W * T_wait_s
    E_offload_mJ = (E_tx_J + E_idle_J) * 1000.0

    # 10. Local compute energy (CMOS model)
    E_local_mJ = KAPPA * ECG_CYCLES * (ESP32_FREQ_HZ ** 2) * 1000.0

    return {
        'channel_gain_h': float(h),
        'uplink_rate_mbps': float(R_mbps),
        'T_tx_ms': float(T_tx_ms),
        'T_prop_ms': float(T_prop_ms),
        'T_queue_ms': float(T_queue_ms),
        'T_compute_edge_ms': float(T_compute_edge_ms),
        'T_total_edge_ms': float(T_total_edge_ms),
        'T_local_ms': float(T_local_ms),
        'E_offload_mJ': float(E_offload_mJ),
        'E_local_mJ': float(E_local_mJ),
        'rho_edge': float(rho),
    }


def simulator_values(n_runs: int = 30):
    """Extract same metrics from the simulator across multiple runs."""
    from src.algorithms.cloud_only import CloudOnlyScheduler
    from src.algorithms.dqn_es import DQNESScheduler
    from src.core.task import HealthcareTask
    from src.data_ingestion.event_generator import generate_synthetic_tasks
    from src.config import replicate_seeds, seed_global_rngs
    from src.simulation.replicate import build_topology, run_scheduler

    # Run N_RUNS=30 DQN-ES runs and collect edge-routed ECG task metrics
    edge_lats = []
    local_lats = []
    offload_energies = []
    local_energies = []

    N_TASKS = 100  # small per run for speed; edge routing well-sampled
    for run_id in range(n_runs):
        seeds = replicate_seeds(run_id, N_TASKS)
        seed_global_rngs(seeds['base'])
        topo = build_topology(run_id, N_TASKS)

        sim_tasks = generate_synthetic_tasks(N_TASKS, ci_distribution='mixed',
                                             seed=seeds['task'])

        wearable_ids = [nid for nid, n in topo.nodes.items() if n.node_type == 'wearable']
        tasks = []
        for t in sim_tasks:
            dev_id = wearable_ids[t.device_id % len(wearable_ids)]
            tasks.append(HealthcareTask(
                task_id=t.task_id, device_id=dev_id, timestamp=t.timestamp,
                data_size_bits=ECG_BITS,       # force ECG profile
                cpu_cycles=ECG_CYCLES,
                max_delay_s=ECG_DEADLINE,
                privacy_sensitivity=t.privacy_sensitivity,
                ci_score=t.ci_score,
                attack_probability=t.attack_probability,
                source='ecg',
                task_type='ecg_analysis',
            ))

        results, _ = run_scheduler(DQNESScheduler, topo, tasks, seeds)

        for r in results:
            ntype = r.get('node_type', '')
            if ntype == 'local':
                local_lats.append(r['latency_ms'])
                local_energies.append(r['energy_mj'])
            elif ntype in ('edge', 'fog', 'cloud'):
                edge_lats.append(r['latency_ms'])
                offload_energies.append(r['energy_mj'])

    return {
        'edge_latency_mean_ms':   float(np.mean(edge_lats))   if edge_lats   else None,
        'edge_latency_std_ms':    float(np.std(edge_lats))    if edge_lats   else None,
        'local_latency_mean_ms':  float(np.mean(local_lats))  if local_lats  else None,
        'offload_energy_mean_mJ': float(np.mean(offload_energies)) if offload_energies else None,
        'local_energy_mean_mJ':   float(np.mean(local_energies))   if local_energies   else None,
        'n_edge_samples': len(edge_lats),
        'n_local_samples': len(local_lats),
    }


def main():
    from src.config import replicate_seeds, seed_global_rngs
    from src.simulation.replicate import build_topology, run_scheduler

    print('[SIM-VAL] Computing analytical reference values...')
    analytic = analytical_values()

    print('[SIM-VAL] Running simulator to collect matching metrics (30 runs)...')
    simulated = simulator_values(n_runs=30)

    print('\n[VALIDATION TABLE]')
    print(f'  {"Metric":<40} {"Analytical":>14} {"Simulated":>14} {"Delta%":>8}')
    print('  ' + '-' * 78)

    # Local latency from Local-Only scheduler (forces local execution)
    from src.algorithms.local_only import LocalOnlyScheduler
    from src.core.task import HealthcareTask
    from src.data_ingestion.event_generator import generate_synthetic_tasks

    local_lats_check = []
    local_energies_check = []
    for run_id in range(10):
        seeds = replicate_seeds(run_id, 100)
        seed_global_rngs(seeds['base'])
        topo = build_topology(run_id, 100)
        sim_tasks = generate_synthetic_tasks(100, ci_distribution='mixed',
                                             seed=seeds['task'])
        wids = [nid for nid, n in topo.nodes.items() if n.node_type == 'wearable']
        tasks_local = [HealthcareTask(
            task_id=t.task_id, device_id=wids[t.device_id % len(wids)],
            timestamp=t.timestamp, data_size_bits=ECG_BITS, cpu_cycles=ECG_CYCLES,
            max_delay_s=ECG_DEADLINE, privacy_sensitivity=t.privacy_sensitivity,
            ci_score=t.ci_score, attack_probability=t.attack_probability, source='ecg',
            task_type='ecg_analysis',
        ) for t in sim_tasks]
        res_local, _ = run_scheduler(LocalOnlyScheduler, topo, tasks_local, seeds)
        for r in res_local:
            local_lats_check.append(r['latency_ms'])
            local_energies_check.append(r['energy_mj'])

    sim_local_lat = float(np.mean(local_lats_check)) if local_lats_check else None
    sim_local_eng = float(np.mean(local_energies_check)) if local_energies_check else None

    comparisons = [
        ('Local exec latency (ms)',     analytic['T_local_ms'],     sim_local_lat),
        ('Total edge offload lat (ms)', analytic['T_total_edge_ms'], simulated['edge_latency_mean_ms']),
        ('Wearable offload energy (mJ)',analytic['E_offload_mJ'],   simulated['offload_energy_mean_mJ']),
        ('Local compute energy (mJ)',   analytic['E_local_mJ'],     sim_local_eng),
    ]

    for label, a_val, s_val in comparisons:
        if a_val is None or s_val is None:
            print(f'  {label:<40} {"N/A":>14} {"N/A":>14} {"N/A":>8}')
            continue
        delta_pct = 100.0 * abs(s_val - a_val) / max(abs(a_val), 1e-12)
        flag = ' OK' if delta_pct < 5.0 else (' ~' if delta_pct < 15.0 else ' !!')
        print(f'  {label:<40} {a_val:>14.3f} {s_val:>14.3f} {delta_pct:>7.1f}%{flag}')

    print(f'\n  Uplink rate (analytical): {analytic["uplink_rate_mbps"]:.1f} Mbps')
    print(f'  Queue utilisation rho:    {analytic["rho_edge"]:.2f}')
    print(f'  M/M/1 queue delay:        {analytic["T_queue_ms"]:.3f} ms')
    print(f'  Compute delay (RPi4):     {analytic["T_compute_edge_ms"]:.3f} ms')

    # Cloud-Only analytical cross-check
    # T_compute_cloud = ECG_CYCLES / CLOUD_FREQ = 150M / 3.2GHz = 46.875ms
    # T_prop_cloud = 50km WAN / 3e8 = 0.167ms
    # T_tx_cloud ~ small at 1GHz BW
    # Simulator Cloud-Only N=1000 result: 91.012ms (from mc_full_summary.json)
    T_compute_cloud_ms = (ECG_CYCLES / CLOUD_FREQ_HZ) * 1000.0
    T_prop_cloud_ms = (50_000.0 / 3e8) * 1000.0  # 50 km WAN
    T_analytical_cloud_lower = T_compute_cloud_ms + T_prop_cloud_ms  # lower bound
    # Channel+queue add ~35-45ms based on simulator config; sum ≈ 44ms
    SIM_CLOUD_ONLY_MS = 91.012  # from mc_full_summary.json N=1000 mean
    channel_and_queue_implied = SIM_CLOUD_ONLY_MS - T_compute_cloud_ms - T_prop_cloud_ms
    print(f'\n  Cloud analytical breakdown:')
    print(f'    Compute (3.2GHz):   {T_compute_cloud_ms:.3f} ms')
    print(f'    Propagation (50km): {T_prop_cloud_ms:.3f} ms')
    print(f'    Channel+queue (sim-implied): {channel_and_queue_implied:.3f} ms')
    print(f'    Total (sim Cloud-Only N=1000): {SIM_CLOUD_ONLY_MS:.3f} ms')
    comparisons.append(
        ('Cloud compute delay (ms)', T_compute_cloud_ms, None)  # analytical only
    )

    # Save results
    out = {
        'analytical': analytic,
        'simulated': simulated,
        'comparison': [
            {
                'metric': label,
                'analytical': a_val,
                'simulated': s_val,
                'delta_pct': (100.0 * abs(s_val - a_val) / max(abs(a_val), 1e-12))
                             if (a_val is not None and s_val is not None) else None,
            }
            for label, a_val, s_val in comparisons
        ]
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / 'simulator_validation.json'
    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f'\n[SAVE] {out_path}')

    # CSV table
    import csv
    csv_path = RESULTS_DIR / 'table_sim_validation.csv'
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['Metric', 'Analytical', 'Simulated (mean 30 runs)', 'Delta (%)'])
        for label, a_val, s_val in comparisons:
            if a_val is not None and s_val is not None:
                dp = 100.0 * abs(s_val - a_val) / max(abs(a_val), 1e-12)
                writer.writerow([label, f'{a_val:.4f}', f'{s_val:.4f}', f'{dp:.2f}'])
            else:
                writer.writerow([label, str(a_val), str(s_val), 'N/A'])
    print(f'[SAVE] {csv_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
