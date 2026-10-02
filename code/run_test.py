"""Quick integration test: run every registered scheduler (main + experiment
registries) on 500 synthetic tasks.

Uses replicate 0 at N=500 (seeds from src.config.replicate_seeds), so every
scheduler sees the same task stream and topology.
"""
import copy, sys, os
BASE = os.path.dirname(os.path.abspath(__file__))  # points to code/
sys.path.insert(0, BASE)

from src.config import get_registry
from src.simulation.replicate import build_synthetic_replicate, run_scheduler

N_TASKS = 500
seeds, topo, tasks = build_synthetic_replicate(0, N_TASKS, 'mixed')

header = f"{'Algorithm':<18} {'Avg Lat (ms)':>14} {'Avg Energy (mJ)':>16} {'Privacy Risk':>13} {'SLA Viols':>10}"
print(header)
print('-' * 74)

for name, sched_cls in get_registry("all").items():
    results, _ = run_scheduler(sched_cls, topo, copy.deepcopy(tasks), seeds)
    assert len(results) == N_TASKS, name
    avg_lat = sum(x['latency_ms'] for x in results) / len(results)
    avg_eng = sum(x['energy_mj'] for x in results) / len(results)
    avg_priv = sum(x['privacy_risk'] for x in results) / len(results)
    sla_v = sum(1 for x in results if x['sla_violated'])
    row = f"{name:<18} {avg_lat:>14.2f} {avg_eng:>16.4f} {avg_priv:>13.4f} {sla_v:>10}/{len(results)}"
    print(row)

print("\nAll schedulers completed successfully.")
