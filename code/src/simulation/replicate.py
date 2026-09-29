"""
Replicate construction shared by every experiment driver.

One replicate = (run_id, n_tasks).  Its task stream, topology, environment
and scheduler seed all come from `src.config.replicate_seeds`, and the
topology is rebuilt per replicate, so replicate r is the same draw for every
algorithm (paired design) and replicates differ in device/fog placement.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.config import (
    N_FOG_NODES,
    N_WEARABLES,
    make_scheduler,
    replicate_seeds,
    seed_global_rngs,
)
from src.core.network import NetworkTopology
from src.core.task import HealthcareTask
from src.simulation.environment import OffloadingEnvironment
from src.simulation.topology import build_healthcare_topology


def to_healthcare_task(t, topology: NetworkTopology) -> HealthcareTask:
    """Convert an event_generator SimulationTask to a HealthcareTask."""
    wids = [nid for nid, n in topology.nodes.items() if n.node_type == 'wearable']
    return HealthcareTask(
        task_id=t.task_id, device_id=wids[t.device_id % len(wids)],
        timestamp=t.timestamp, data_size_bits=t.data_size_bits,
        cpu_cycles=t.cpu_cycles, max_delay_s=t.max_delay_s,
        privacy_sensitivity=t.privacy_sensitivity, ci_score=t.ci_score,
        attack_probability=t.attack_probability, source=t.source,
        task_type=getattr(t, 'task_type', 'unknown'),
    )


def build_topology(run_id: int, n_tasks: int = 0,
                   n_wearables: int = N_WEARABLES,
                   n_fog_nodes: int = N_FOG_NODES) -> NetworkTopology:
    seeds = replicate_seeds(run_id, n_tasks)
    return build_healthcare_topology(
        n_wearables=n_wearables, n_fog_nodes=n_fog_nodes,
        seed=seeds['topology'],
    )


def build_synthetic_replicate(
    run_id: int,
    n_tasks: int,
    ci_distribution: str = 'mixed',
    n_wearables: int = N_WEARABLES,
    n_fog_nodes: int = N_FOG_NODES,
) -> Tuple[dict, NetworkTopology, List[HealthcareTask]]:
    """Seeds, per-replicate topology and synthetic task stream."""
    from src.data_ingestion.event_generator import generate_synthetic_tasks
    seeds = replicate_seeds(run_id, n_tasks)
    seed_global_rngs(seeds['base'])
    topo = build_topology(run_id, n_tasks, n_wearables, n_fog_nodes)
    sim_tasks = generate_synthetic_tasks(n_tasks, ci_distribution,
                                         seed=seeds['task'])
    tasks = [to_healthcare_task(t, topo) for t in sim_tasks]
    return seeds, topo, tasks


def run_scheduler(
    sched_cls,
    topology: NetworkTopology,
    tasks: List[HealthcareTask],
    seeds: dict,
    sched_kwargs: Optional[dict] = None,
):
    """Build the scheduler with the replicate seed and run the episode."""
    sched = make_scheduler(sched_cls, topology, seeds['scheduler'],
                           **(sched_kwargs or {}))
    env = OffloadingEnvironment(topology, sched, n_tasks=len(tasks),
                                seed=seeds['env'])
    results = env.run(tasks)
    return results, sched
