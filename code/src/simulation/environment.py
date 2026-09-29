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

from src.core.cost_function import compute_privacy_risk
from src.core.offload_model import offload_outcome
from src.core.network import NetworkTopology
from src.core.task import HealthcareTask
from src.algorithms.base_scheduler import BaseScheduler
from src.config import ATTACK_BURST_INTENSITY, ATTACK_BURST_PROB, ci_tier
from src.core.offload_model import result_size_bits


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
        self._device_decisions: Dict[int, int] = {}   # decisions made per device

        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reinitialise all mutable state."""
        for wid in self._wearable_ids:
            self._battery_j[wid] = BATTERY_CAPACITY_J
        self._device_decisions = {}

        self.topology.reset_queues()

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

        # --- Advance the clock: release tasks finished by now (D16) ---
        self.topology.advance_time(task.timestamp)

        # --- Schedule (Fix C: record wall-clock time for select_node) ---
        if hasattr(self.scheduler, 'last_decision_info'):
            self.scheduler.last_decision_info = {}
        _t0 = time.perf_counter()
        node_id = self.scheduler.select_node(task)
        scheduling_overhead_ms = (time.perf_counter() - _t0) * 1000.0
        task.assigned_node = node_id
        info = getattr(self.scheduler, 'last_decision_info', None) or {}
        dev_idx = self._device_decisions.get(task.device_id, 0)
        self._device_decisions[task.device_id] = dev_idx + 1

        # --- Retrieve node info ---
        dst_node = self.topology.get_node(node_id)
        src_node = self.topology.get_node(task.device_id)
        is_local = (node_id == task.device_id)

        # --- Latency / energy with component breakdown (core/offload_model) ---
        # Latency is capped at 999 s (overloaded queue) before the idle-energy
        # term is computed, as in round 1.
        out = offload_outcome(self.topology, task, node_id, latency_cap_s=999.0,
                              realised=True)
        latency_s = out.latency_s
        energy_j = max(0.0, out.energy_j)
        lat_tx_s, lat_prop_s = out.t_tx, out.t_prop
        lat_queue_s, lat_comp_s, lat_dl_s = out.t_queue, out.t_proc, out.t_dl

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

        # --- Queue update: the task holds its node until service ends (D16) ---
        wait, _done = self.topology.reserve(node_id, out.t_arrive, out.t_proc)
        if abs(wait - out.t_queue) > 1e-9:
            raise RuntimeError('realised FIFO wait differs from the prediction')

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
            'latency_downlink_ms': lat_dl_s    * 1000.0,  # E9: return prop + result download
            'energy_mj':           energy_j * 1000.0,
            'energy_tx_mj':        out.e_tx * 1000.0,
            'energy_idle_mj':      out.e_idle * 1000.0,
            'energy_rx_mj':        out.e_rx * 1000.0,
            'energy_compute_mj':   out.e_compute * 1000.0,
            'privacy_risk':        privacy_risk,
            'cost':                cost,
            'sla_violated':        sla_violated,
            'sla_deadline_ms':     task.max_delay_s * 1000.0,
            # True Phi (label); ci_visible is what the scheduler saw (E14)
            'ci_score':            task.ci_score if task.ci_true is None else task.ci_true,
            'ci_visible':          task.ci_score,
            'attack_prob':         task.attack_probability,
            'battery_remaining_j': self._battery_j.get(task.device_id, -1.0),
            'timestamp':           task.timestamp,
            'scheduling_overhead_ms': scheduling_overhead_ms,  # Fix C: timing
            # Raw-log fields (plan E5, E8, E11; simulation/episode_log.py)
            'task_type':           task.task_type,
            'ci_tier':             ci_tier(task.ci_score if task.ci_true is None
                                           else task.ci_true),
            'payload_bits':        task.data_size_bits,
            'result_bits':         0 if is_local else result_size_bits(task),
            'completion_time_s':   task.timestamp + latency_s,
            'device_decision_index': dev_idx,   # 0-based, per device
            'explored':            info.get('explored'),
            'epsilon':             info.get('epsilon'),
            'q_argmax_node':       info.get('q_argmax_node'),
            'exec_q_rank':         info.get('exec_q_rank'),
            't_bounds_ms':         info.get('t_bounds_ms'),
            't_forward_ms':        info.get('t_forward_ms'),
            't_enum_ms':           info.get('t_enum_ms'),
            't_update_ms':         info.get('t_update_ms'),
            'td_loss':             info.get('td_loss'),
            'q_max':               info.get('q_max'),
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

            # Occupancy is released at completion time by advance_time().

        return results


    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_battery_levels(self) -> Dict[int, float]:
        """Return current battery (Joules) per wearable device."""
        return dict(self._battery_j)

    def get_queue_loads(self) -> Dict[int, int]:
        """Return current queue task count per node."""
        return {nid: n.current_load for nid, n in self.topology.nodes.items()}

