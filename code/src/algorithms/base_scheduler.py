"""
Abstract base class for all task-offloading schedulers.

Every concrete scheduler must implement `select_node(task)` and may
optionally override `evaluate_node` if it needs custom cost components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Optional, Tuple

from src.core.cost_function import (
    compute_cost,
    compute_privacy_risk,
    estimate_bounds,
)
from src.core.offload_model import offload_outcome
from src.core.network import NetworkTopology
from src.core.task import HealthcareTask

# W = 50: sliding window size for routing-entropy privacy estimation.
# Spans approximately five CI cycles; Section 3.4 and simulation setup.
WINDOW_SIZE: int = 50


class BaseScheduler(ABC):
    """
    Abstract base for all scheduling algorithms.

    Parameters
    ----------
    topology : NetworkTopology
        The current network graph.
    offload_history : dict, optional
        Per-device sliding-window history of offloading decisions.
        Structure: {device_id: deque([node_id, node_id, ...])} with maxlen=WINDOW_SIZE.
        Legacy dict-of-counts input is accepted and converted automatically.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        offload_history: Optional[Dict] = None,
    ):
        self.topology = topology
        # FIX C6: use deque(maxlen=WINDOW_SIZE) per device for sliding-window entropy.
        # Accept either legacy dict-of-counts or new deque format.
        self.offload_history: Dict[int, deque] = {}
        if offload_history:
            for dev_id, hist in offload_history.items():
                d: deque = deque(maxlen=WINDOW_SIZE)
                if isinstance(hist, dict):
                    # Convert legacy {node_id: count} to a flat deque
                    for node_id, cnt in hist.items():
                        d.extend([node_id] * min(cnt, WINDOW_SIZE))
                elif hasattr(hist, '__iter__'):
                    d.extend(hist)
                self.offload_history[dev_id] = d

        # Pre-compute sorted list of all non-wearable candidate node IDs
        self._candidate_nodes = sorted(
            n.node_id
            for n in topology.nodes.values()
            if n.node_type != 'wearable'
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def select_node(self, task: HealthcareTask) -> int:
        """
        Return the node_id to which this task should be assigned.
        Must be implemented by every concrete scheduler.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared evaluation helper
    # ------------------------------------------------------------------

    def evaluate_node(
        self,
        task: HealthcareTask,
        node_id: int,
        latency_bounds: Tuple[float, float],
        energy_bounds: Tuple[float, float],
    ) -> Tuple[float, float, float, float]:
        """
        Compute (cost, latency_s, energy_j, privacy_risk) for assigning
        *task* to *node_id*.

        For the wearable's own node (local execution):
          - Latency  = C_i / f_local
          - Energy   = Îº Â· C_i Â· fÂ²
          - Privacy  = 0  (data never leaves the device)

        For remote nodes (core/offload_model.py):
          - Latency  = t_tx + t_prop + t_queue + t_proc + t_dl
          - Energy   = P_tx Â· t_tx + P_rx Â· t_rx + P_idle Â· (L - t_tx - t_rx)
          - Privacy  = Ï Â· (1 - H / H_max)

        Parameters
        ----------
        task           : HealthcareTask
        node_id        : int â€” target node
        latency_bounds : (l_min, l_max) in seconds   â€” for normalisation
        energy_bounds  : (e_min, e_max) in joules    â€” for normalisation

        Returns
        -------
        (cost, latency_s, energy_j, privacy_risk)
        """
        outcome = offload_outcome(self.topology, task, node_id)
        latency_s = outcome.latency_s
        energy_j = outcome.energy_j
        is_local = outcome.is_local

        # ---- Privacy risk ----
        if is_local:
            privacy_risk = 0.0
        else:
            device_deque = self.offload_history.get(task.device_id, deque())
            counts: Dict[int, int] = {}
            for nid in device_deque:
                if nid != task.device_id:  # Exclude local execution from network entropy
                    counts[nid] = counts.get(nid, 0) + 1
            
            # PROSPECTIVE FIX: add the node_id being evaluated to the counts
            counts[node_id] = counts.get(node_id, 0) + 1

            n_available = len(self._candidate_nodes)
            privacy_risk = compute_privacy_risk(
                task.privacy_sensitivity,
                counts,
                n_available,
            )

        # ---- Composite cost ----
        cost = compute_cost(
            latency_s,
            energy_j,
            privacy_risk,
            task.ci_score,
            latency_bounds,
            energy_bounds,
        )

        return cost, latency_s, energy_j, privacy_risk

    # ------------------------------------------------------------------
    # Shared bound estimation
    # ------------------------------------------------------------------

    def estimate_feasible_bounds(
        self,
        task: HealthcareTask,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Quick sweep over all candidate nodes to get min/max latency &
        energy â€” used as normalisation bounds in compute_cost.
        """
        latencies: list[float] = []
        energies: list[float] = []

        # Local execution plus every remote candidate
        for nid in [task.device_id] + list(self._candidate_nodes):
            try:
                out = offload_outcome(self.topology, task, nid)
            except Exception:
                continue
            if math.isfinite(out.latency_s) and math.isfinite(out.energy_j):
                latencies.append(out.latency_s)
                energies.append(out.energy_j)

        return estimate_bounds(latencies, energies)

    # ------------------------------------------------------------------
    # History tracking
    # ------------------------------------------------------------------

    def record_decision(self, device_id: int, node_id: int) -> None:
        """Update the sliding-window offloading history after a scheduling decision."""
        if device_id not in self.offload_history:
            self.offload_history[device_id] = deque(maxlen=WINDOW_SIZE)
        self.offload_history[device_id].append(node_id)

    @property
    def candidate_nodes(self):
        return self._candidate_nodes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nodes={len(self._candidate_nodes)})"


import math  # noqa: E402 â€” placed at bottom to avoid circular import issues

