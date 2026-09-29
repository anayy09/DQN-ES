"""
Simulation environment for the IoT-Edge-Cloud task offloading system.

The environment manages one round of scheduling decisions, tracking:
  - Network queue loads (arrival rates updated each step)
  - Per-task metrics (latency, energy, SLA, privacy risk)
  - Attack probability injection (adversarial scenario)
  - Battery depletion (wearable energy budget)

Usage:
    env = OffloadingEnvironment(topology, scheduler, n_tasks=1000, seed=42)
    env.reset()
    results = env.run(tasks)
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional

import numpy as np

from src.core.cost_function import (
    compute_local_energy,
    compute_local_latency,
    compute_offload_energy,
    compute_offload_latency,
    compute_privacy_risk,
    estimate_bounds,
)
from src.core.network import NetworkTopology
from src.core.task import HealthcareTask
from src.algorithms.base_scheduler import BaseScheduler
from src.config import ATTACK_BURST_INTENSITY, ATTACK_BURST_PROB


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATTERY_CAPACITY_J = 0.5 * 3600.0   # 500 mAh @ 3.7 V â‰ˆ 1850 J per wearable


class OffloadingEnvironment:
    """
    Discrete-event simulation environment.

    Parameters
    ----------
    topology    : NetworkTopology  â€” network graph with all nodes and links
    scheduler   : BaseScheduler    â€” the algorithm under test
    n_tasks     : int              â€” total tasks in simulation (informational)
    seed        : int              â€” RNG seed for reproducibility
    """

    def __init__(
        self,
        topology: NetworkTopology,
        scheduler: BaseScheduler,
        n_tasks: int = 1000,
        seed: int = 42,
        attack_burst_prob: Optional[float] = None,
        attack_burst_intensity: Optional[float] = None,
    ):
        self.topology = topology
        self.attack_burst_prob = (ATTACK_BURST_PROB if attack_burst_prob is None
                                  else attack_burst_prob)
        self.attack_burst_intensity = (ATTACK_BURST_INTENSITY
                                       if attack_burst_intensity is None
                                       else attack_burst_intensity)
        self.scheduler = scheduler
        self.n_tasks = n_tasks
        self.seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Identify node categories
        self._wearable_ids = [
            nid for nid, n in topology.nodes.items() if n.node_type == 'wearable'
        ]
        self._compute_ids = [
            nid for nid, n in topology.nodes.items() if n.node_type != 'wearable'
        ]
        self._cloud_id = max(
            (nid for nid, n in topology.nodes.items() if n.node_type == 'cloud'),
            default=max(topology.nodes.keys()),
        )

        # State variables (reset on reset())
        self._battery_j: Dict[int, float] = {}
        self._queue_task_counts: Dict[int, int] = {}
        self._time_window_s: float = 1.0      # sliding window for Î» estimation
        self._recent_arrivals: Dict[int, List[float]] = {}   # node_id â†’ arrival times

        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reinitialise all mutable state."""
        for wid in self._wearable_ids:
            self._battery_j[wid] = BATTERY_CAPACITY_J

        for nid in self.topology.nodes:
            self._queue_task_counts[nid] = 0
            self.topology.nodes[nid].current_load = 0
            self.topology.nodes[nid].arrival_rate = 0.0
            self._recent_arrivals[nid] = []

    # ------------------------------------------------------------------
    # Single task step
    # ------------------------------------------------------------------

    def step(self, task: HealthcareTask) -> dict:
        """
        Execute one task through the scheduler and compute all metrics.

        Parameters
        ----------
        task : HealthcareTask

        Returns
        -------
        metrics : dict with keys:
          task_id, device_id, assigned_node, node_type,
          latency_ms, energy_mj, privacy_risk, cost,
          sla_violated, battery_remaining_j, timestamp
        """
        # --- Inject attack probability (adversarial scenario) ---
        if self._rng.random() < self.attack_burst_prob:
            task.attack_probability = self.attack_burst_intensity
        else:
            task.attack_probability = max(task.attack_probability, 0.0)

        # --- Update node queue loads before scheduling ---
        self._update_arrival_rates(task.timestamp)

        # --- Schedule (Fix C: record wall-clock time for select_node) ---
        _t0 = time.perf_counter()
        node_id = self.scheduler.select_node(task)
        scheduling_overhead_ms = (time.perf_counter() - _t0) * 1000.0
        task.assigned_node = node_id

        # --- Retrieve node info ---
        dst_node = self.topology.get_node(node_id)
        src_node = self.topology.get_node(task.device_id)
        is_local = (node_id == task.device_id)

        # --- Compute latency and component breakdown (Fix F) ---
        if is_local:
            latency_s = compute_local_latency(
                task.cpu_cycles, src_node.hardware.cpu_freq_hz
            )
            # Component breakdown for local execution
            lat_tx_s    = 0.0
            lat_prop_s  = 0.0
            lat_queue_s = 0.0
            lat_comp_s  = latency_s
        else:
            uplink_rate = self.topology.get_uplink_rate(task.device_id, node_id)
            prop_delay = self.topology.get_propagation_delay(task.device_id, node_id)
            queue_delay = self.topology.get_queue_delay(node_id)
            latency_s = compute_offload_latency(
                task.data_size_bits,
                task.cpu_cycles,
                uplink_rate,
                prop_delay,
                queue_delay,
                dst_node.hardware.cpu_freq_hz,
            )
            # Component breakdown: t_tx + t_prop + t_queue + t_proc
            lat_tx_s    = task.data_size_bits / max(uplink_rate, 1.0)
            lat_prop_s  = prop_delay
            lat_queue_s = queue_delay
            lat_comp_s  = task.cpu_cycles / max(dst_node.hardware.cpu_freq_hz, 1.0)

        # Cap latency at a sensible maximum (prevents infinity propagation)
        latency_s = min(latency_s, 999.0)

        # --- Compute energy ---
        if is_local:
            energy_j = compute_local_energy(
                task.cpu_cycles,
                src_node.hardware.cpu_freq_hz,
                src_node.hardware.kappa,
            )
        else:
            uplink_rate = self.topology.get_uplink_rate(task.device_id, node_id)
            energy_j = compute_offload_energy(
                task.data_size_bits,
                uplink_rate,
                latency_s,
                src_node.hardware.tx_power_w,
                src_node.hardware.idle_power_w,
            )
        energy_j = max(0.0, energy_j)

        # --- Privacy risk ---
        if is_local:
            privacy_risk = 0.0
        else:
            device_deque = self.scheduler.offload_history.get(task.device_id, [])
            device_history: Dict[int, int] = {}
            for nid in device_deque:
                device_history[nid] = device_history.get(nid, 0) + 1
            n_available = len(self._compute_ids)
            privacy_risk = compute_privacy_risk(
                task.privacy_sensitivity,
                device_history,
                n_available,
            )

        # --- SLA check ---
        sla_violated = latency_s > task.max_delay_s
        task.actual_latency_s = latency_s
        task.actual_energy_j = energy_j
        task.sla_violated = sla_violated

        # --- Battery update ---
        if task.device_id in self._battery_j:
            self._battery_j[task.device_id] = max(
                0.0,
                self._battery_j[task.device_id] - energy_j,
            )

        # --- Queue update: mark task arrival at destination node ---
        self._queue_task_counts[node_id] = self._queue_task_counts.get(node_id, 0) + 1
        self._recent_arrivals[node_id].append(task.timestamp)
        self.topology.update_load(node_id, delta=+1)

        # --- Compute composite cost for diagnostics ---
        lat_bounds = (0.0, max(latency_s * 2.0, 1e-3))
        eng_bounds = (0.0, max(energy_j * 2.0, 1e-12))
        from src.core.cost_function import compute_cost
        cost = compute_cost(
            latency_s, energy_j, privacy_risk,
            task.ci_score, lat_bounds, eng_bounds,
        )

        return {
            'task_id':             task.task_id,
            'device_id':           task.device_id,
            'assigned_node':       node_id,
            'node_type':           dst_node.node_type,
            'latency_ms':          latency_s * 1000.0,
            'latency_tx_ms':       lat_tx_s   * 1000.0,   # Fix F: component
            'latency_prop_ms':     lat_prop_s  * 1000.0,  # Fix F: component
            'latency_queue_ms':    lat_queue_s * 1000.0,  # Fix F: component
            'latency_compute_ms':  lat_comp_s  * 1000.0,  # Fix F: component
            'energy_mj':           energy_j * 1000.0,
            'privacy_risk':        privacy_risk,
            'cost':                cost,
            'sla_violated':        sla_violated,
            'sla_deadline_ms':     task.max_delay_s * 1000.0,
            'ci_score':            task.ci_score,
            'attack_prob':         task.attack_probability,
            'battery_remaining_j': self._battery_j.get(task.device_id, -1.0),
            'timestamp':           task.timestamp,
            'scheduling_overhead_ms': scheduling_overhead_ms,  # Fix C: timing
        }

    # ------------------------------------------------------------------
    # Run all tasks
    # ------------------------------------------------------------------

    def run(self, tasks: List[HealthcareTask]) -> List[dict]:
        """
        Process all tasks in arrival-time order.

        Parameters
        ----------
        tasks : list of HealthcareTask (sorted by timestamp)

        Returns
        -------
        results : list of per-task metric dicts
        """
        # Sort by arrival time to respect causality
        tasks_sorted = sorted(tasks, key=lambda t: t.timestamp)
        results = []

        for task in tasks_sorted:
            metrics = self.step(task)
            results.append(metrics)

            # Periodically release completed tasks from queue
            # (simple model: task completes after its latency)
            complete_time = task.timestamp + (metrics['latency_ms'] / 1000.0)
            assigned_node = metrics['assigned_node']
            if self.topology.nodes[assigned_node].current_load > 0:
                self.topology.update_load(assigned_node, delta=-1)

        return results

    # ------------------------------------------------------------------
    # Arrival rate estimation (sliding window)
    # ------------------------------------------------------------------

    def _update_arrival_rates(self, current_time: float) -> None:
        """
        Update arrival rates Î» for each node using a sliding time window.
        Prune arrivals older than _time_window_s from the history.
        """
        for nid in self.topology.nodes:
            # Prune stale arrivals
            cutoff = current_time - self._time_window_s
            arrivals = [t for t in self._recent_arrivals.get(nid, []) if t > cutoff]
            self._recent_arrivals[nid] = arrivals

            # Î» = count / window_s
            lam = len(arrivals) / self._time_window_s
            self.topology.update_arrival_rate(nid, lam)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_battery_levels(self) -> Dict[int, float]:
        """Return current battery (Joules) per wearable device."""
        return dict(self._battery_j)

    def get_queue_loads(self) -> Dict[int, int]:
        """Return current queue task count per node."""
        return {nid: n.current_load for nid, n in self.topology.nodes.items()}

