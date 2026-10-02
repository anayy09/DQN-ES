"""
ES-only ablation scheduler (replaces ES-only).

Identical to DQN-ES except the DQN top-K pre-filter is replaced by a 
full exhaustive search over the entire candidate-node set.

This baseline isolates the contribution of the DQN action-space compression
from the exact exhaustive evaluation.
"""

from __future__ import annotations

import time
from typing import List, Optional

from src.algorithms.base_scheduler import BaseScheduler
from src.core.task import HealthcareTask


class ESOnlyScheduler(BaseScheduler):
    """
    Pure Exhaustive Search over the full candidate-node set, no DQN.

    privacy_weight_scale (lambda_P) multiplies the CI-adaptive privacy weight
    before renormalisation; sweeping it traces the latency-privacy frontier
    reachable by a myopic reweighted greedy without any learning.
    """

    def __init__(
        self,
        topology,
        seed: int = 42,
        offload_history: Optional[dict] = None,
        privacy_weight_scale: float = 1.0,
    ):
        super().__init__(topology, offload_history)
        if privacy_weight_scale <= 0:
            raise ValueError('privacy_weight_scale must be > 0')
        # lambda_P > 1 gives the reweighted-greedy baseline
        self.privacy_weight_scale = float(privacy_weight_scale)
        self._idx_to_node: List[int] = self._candidate_nodes
        self._n_nodes = len(self._idx_to_node)
        self.dispatch_times_ms: List[float] = []
        self.last_decision_info: dict = {}

    def select_node(self, task: HealthcareTask) -> int:
        if self._n_nodes == 1:
            nid = self._idx_to_node[0]
            self.record_decision(task.device_id, nid)
            return nid

        t0 = time.perf_counter()
        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)
        t_start = time.perf_counter()
        
        best_node = -1
        best_cost = float('inf')
        
        for node_id in self._idx_to_node:
            cost, _, _, _ = self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
            if cost < best_cost:
                best_cost = cost
                best_node = node_id
                
        t_end = time.perf_counter()
        self.dispatch_times_ms.append((t_end - t_start) * 1000.0)
        self.last_decision_info = {
            't_bounds_ms': (t_start - t0) * 1000.0,
            't_forward_ms': 0.0,
            't_enum_ms': (t_end - t_start) * 1000.0,
            't_update_ms': 0.0,
        }

        self.record_decision(task.device_id, best_node)
        return best_node
