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


# Per-task knobs (applied after generation) and workload knobs (applied at
# generation).  Other condition keys (warm start, CI noise) are added below.
TASK_OVERRIDE_KEYS = ('ecg_payload_bits', 'result_size_bits')
WORKLOAD_KEYS = ('arrival_process', 'load_rho')
SCHEDULER_KEYS = ('warm_start_tasks',)
CI_NOISE_KEYS = ('ci_noise_sigma', 'ci_misclass_prob')


def perturb_ci(tasks: List[HealthcareTask], sigma: Optional[float],
               misclass_prob: Optional[float], seed: int) -> None:
    """
    E14: replace each task's scheduler-visible ci_score by a perturbed value
    and keep the true value in ci_true.  With probability misclass_prob the
    value is redrawn uniformly inside one of the other two CI tiers; then
    Gaussian noise N(0, sigma^2) is added and the result clipped to [0, 1].
    """
    import numpy as np
    from src.config import CI_TIER_BOUNDS
    lo, hi = CI_TIER_BOUNDS
    tiers = [(0.0, lo), (lo, hi), (hi, 1.0)]
    rng = np.random.default_rng(seed)
    for t in tasks:
        true = t.ci_score if t.ci_true is None else t.ci_true
        v = true
        # Draw both variates for every task so the noise sequence does not
        # depend on which options are active.
        u_mis, u_tier, u_val, z = rng.random(), rng.random(), rng.random(), rng.normal()
        if misclass_prob and u_mis < misclass_prob:
            cur = 0 if true < lo else (1 if true < hi else 2)
            other = [i for i in range(3) if i != cur][int(u_tier * 2)]
            a, b = tiers[other]
            v = a + u_val * (b - a)
        if sigma:
            v = v + sigma * z
        t.ci_true = true
        t.ci_score = float(min(1.0, max(0.0, v)))


def apply_task_overrides(tasks: List[HealthcareTask],
                         overrides: Optional[dict],
                         seeds: Optional[dict] = None) -> List[HealthcareTask]:
    """
    Sensitivity knobs applied to a built task list (in place):
      ecg_payload_bits   D_i of every ECG task (D15: 10 KB sensitivity)
      result_size_bits   result returned to the wearable (E9: 1-64 KB)
    """
    if not overrides:
        return tasks
    unknown = (set(overrides) - set(TASK_OVERRIDE_KEYS) - set(WORKLOAD_KEYS)
               - set(SCHEDULER_KEYS) - set(CI_NOISE_KEYS))
    if unknown:
        raise KeyError(f'unknown task overrides: {sorted(unknown)}')
    for t in tasks:
        if overrides.get('ecg_payload_bits') is not None and t.task_type == 'ecg_analysis':
            t.data_size_bits = int(overrides['ecg_payload_bits'])
        if overrides.get('result_size_bits') is not None:
            t.result_size_bits = int(overrides['result_size_bits'])
    if overrides.get('ci_noise_sigma') or overrides.get('ci_misclass_prob'):
        if seeds is None:
            raise ValueError('CI noise needs the replicate seeds')
        perturb_ci(tasks, overrides.get('ci_noise_sigma'),
                   overrides.get('ci_misclass_prob'), seeds['ci_noise'])
    return tasks


def overrides_tag(overrides: Optional[dict]) -> str:
    """Short label for output paths, e.g. 'ecg80000_res8000'."""
    if not overrides:
        return ''
    parts = []
    if overrides.get('ecg_payload_bits') is not None:
        parts.append(f"ecg{int(overrides['ecg_payload_bits'])}")
    if overrides.get('result_size_bits') is not None:
        parts.append(f"res{int(overrides['result_size_bits'])}")
    if overrides.get('arrival_process', 'poisson') != 'poisson':
        parts.append(str(overrides['arrival_process']))
    if overrides.get('load_rho') is not None:
        parts.append(f"rho{overrides['load_rho']:g}")
    if overrides.get('warm_start_tasks'):
        parts.append(f"warm{int(overrides['warm_start_tasks'])}")
    if overrides.get('ci_noise_sigma'):
        parts.append(f"cin{overrides['ci_noise_sigma']:g}")
    if overrides.get('ci_misclass_prob'):
        parts.append(f"cim{overrides['ci_misclass_prob']:g}")
    return '_'.join(parts)


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
    task_overrides: Optional[dict] = None,
) -> Tuple[dict, NetworkTopology, List[HealthcareTask]]:
    """Seeds, per-replicate topology and synthetic task stream."""
    from src.data_ingestion.event_generator import generate_synthetic_tasks
    seeds = replicate_seeds(run_id, n_tasks)
    seed_global_rngs(seeds['base'])
    topo = build_topology(run_id, n_tasks, n_wearables, n_fog_nodes)
    ov = task_overrides or {}
    rate = None
    if ov.get('load_rho') is not None:
        from src.data_ingestion.event_generator import arrival_rate_for_edge_load
        rate = arrival_rate_for_edge_load(float(ov['load_rho']))
    sim_tasks = generate_synthetic_tasks(
        n_tasks, ci_distribution, seed=seeds['task'],
        arrival_process=ov.get('arrival_process', 'poisson'),
        arrival_rate=rate, arrival_seed=seeds['arrival'])
    tasks = [to_healthcare_task(t, topo) for t in sim_tasks]
    apply_task_overrides(tasks, task_overrides, seeds)
    return seeds, topo, tasks


def warm_start(sched, topology: NetworkTopology, seeds: dict,
               n_pre: int) -> bool:
    """
    Pre-train a DQN scheduler (plan E12) on n_pre synthetic tasks drawn from
    the replicate's 'pretrain' seed stream (disjoint from every evaluation
    stream), on the same topology.  Network weights, replay buffer and the
    decayed epsilon carry over; queues, routing history, the pending
    transition and per-episode logs are reset.  Returns False (no-op) for
    schedulers without a learner.
    """
    if not n_pre or not hasattr(sched, '_online_net'):
        return False
    from src.data_ingestion.event_generator import generate_synthetic_tasks
    pre_seed = seeds['pretrain']
    pre = [to_healthcare_task(t, topology) for t in
           generate_synthetic_tasks(int(n_pre), 'mixed', seed=pre_seed)]
    OffloadingEnvironment(topology, sched, n_tasks=len(pre),
                          seed=pre_seed + 1).run(pre)
    sched.offload_history = {}
    sched._pending = None
    for attr in ('epsilon_history', 'dispatch_times_ms'):
        if hasattr(sched, attr):
            setattr(sched, attr, [])
    sched.warm_started_with = int(n_pre)
    return True


def run_scheduler(
    sched_cls,
    topology: NetworkTopology,
    tasks: List[HealthcareTask],
    seeds: dict,
    sched_kwargs: Optional[dict] = None,
    warm_start_tasks: Optional[int] = None,
):
    """Build the scheduler with the replicate seed and run the episode."""
    sched = make_scheduler(sched_cls, topology, seeds['scheduler'],
                           **(sched_kwargs or {}))
    if warm_start_tasks:
        warm_start(sched, topology, seeds, warm_start_tasks)
    env = OffloadingEnvironment(topology, sched, n_tasks=len(tasks),
                                seed=seeds['env'])
    results = env.run(tasks)
    return results, sched
