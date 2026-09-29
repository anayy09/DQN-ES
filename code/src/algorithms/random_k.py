"""
Randomised candidate-restriction controls (plan E2; self-review F3).

RandomKScheduler   a uniformly random K-subset of the network destinations
                   each task, then argmin F over it.  No learning: this is
                   DQN-ES with epsilon fixed at 1 and no Q-network.
QMixedScheduler    with probability q a random K-subset, otherwise full
                   enumeration (ES-only).  q = 0 is ES-only and q = 1 is
                   RandomKScheduler, decision for decision, so a q-sweep
                   traces the latency-privacy frontier of cheap randomisation.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.config import DQN_TOP_K, GLOBAL_SEED
from src.core.task import HealthcareTask


class RandomKScheduler(BaseScheduler):
    """Uniform random K-subset, then exhaustive enumeration over it."""

    def __init__(
        self,
        topology,
        n_candidate_nodes: int = DQN_TOP_K,
        seed: int = GLOBAL_SEED,
        offload_history: Optional[dict] = None,
    ):
        super().__init__(topology, offload_history)
        self._idx_to_node: List[int] = self._candidate_nodes
        self._n_nodes = len(self._idx_to_node)
        if not 1 <= n_candidate_nodes <= self._n_nodes:
            raise ValueError(
                f'K must be in 1..{self._n_nodes}, got {n_candidate_nodes}')
        self.n_candidate_nodes = n_candidate_nodes
        self._rng = np.random.default_rng(seed)
        self.dispatch_times_ms: List[float] = []
        # Per-decision diagnostics (read by the environment logger)
        self.last_decision_info: dict = {}

    def _use_random_subset(self) -> bool:
        return True

    def select_node(self, task: HealthcareTask) -> int:
        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)
        t0 = time.perf_counter()
        explored = self._use_random_subset()
        if explored:
            cand = self._rng.choice(self._n_nodes, size=self.n_candidate_nodes,
                                    replace=False).tolist()
        else:
            cand = list(range(self._n_nodes))

        best_node, best_cost = -1, float('inf')
        for idx in cand:
            node_id = self._idx_to_node[idx]
            cost, _, _, _ = self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
            if cost < best_cost:
                best_cost, best_node = cost, node_id
        self.dispatch_times_ms.append((time.perf_counter() - t0) * 1000.0)

        self.last_decision_info = {
            'explored': bool(explored),
            'candidates': [self._idx_to_node[i] for i in cand],
        }
        self.record_decision(task.device_id, best_node)
        return best_node


class QMixedScheduler(RandomKScheduler):
    """With probability q a random K-subset, otherwise full enumeration."""

    def __init__(
        self,
        topology,
        q: float = 0.5,
        n_candidate_nodes: int = DQN_TOP_K,
        seed: int = GLOBAL_SEED,
        offload_history: Optional[dict] = None,
    ):
        if not 0.0 <= q <= 1.0:
            raise ValueError(f'q must be in [0, 1], got {q}')
        super().__init__(topology, n_candidate_nodes=n_candidate_nodes,
                         seed=seed, offload_history=offload_history)
        self.q = float(q)

    def _use_random_subset(self) -> bool:
        # No draw at the endpoints, so q=0 == ES-only and q=1 == Random-K
        # decision for decision under the same seed.
        if self.q <= 0.0:
            return False
        if self.q >= 1.0:
            return True
        return bool(self._rng.random() < self.q)
