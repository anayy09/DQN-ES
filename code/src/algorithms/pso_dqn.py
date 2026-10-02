"""
PSO+DQN Scheduler — attribution control for DQN-ES.

Identical to DQNESScheduler except the inner search over the K-node
candidate subspace uses PSO instead of exhaustive search.

This pairing tests whether the DRL top-K pre-filter structure, rather
than the specific inner-search algorithm (exhaustive vs PSO),
drives the privacy advantage.

At K=3, PSO with standard hyperparameters exhaustively evaluates all
three candidates regardless of swarm size, so PSO+DQN ≈ DQN-ES is
expected. See Section 5.4 and the PSO+DQN attribution analysis.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.algorithms.dqn_es import DQNESScheduler
from src.config import DQN_TOP_K
from src.core.task import HealthcareTask


class PSODQNScheduler(DQNESScheduler):
    """
    DQN top-K pre-filter + PSO inner search over the K-node subspace.

    All components identical to DQNESScheduler:
      - State representation (Eq. 27)
      - CI-modulated reward (Eq. 28)
      - Bellman update (Eq. 30)
      - Epsilon-greedy top-K selection

    The only structural difference: _exhaustive_search is replaced by
    _pso_inner_search, which runs PSO restricted to candidate_indices.

    PSO hyperparameters (matching standalone PSO baseline):
      n_particles = 30, max_iter = 50, w = 0.7, c1 = 1.5, c2 = 1.5
    """

    def __init__(
        self,
        topology,
        n_candidate_nodes: int = DQN_TOP_K,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        gamma: float = 0.95,
        lr: float = 0.001,
        replay_capacity: int = 10_000,
        batch_size: int = 32,
        target_sync_freq: int = 50,
        seed: int = 42,
        offload_history: Optional[dict] = None,
        # PSO inner search params (match standalone PSO baseline)
        n_particles: int = 30,
        max_iter: int = 50,
        inertia: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ):
        super().__init__(
            topology=topology,
            n_candidate_nodes=n_candidate_nodes,
            epsilon=epsilon,
            epsilon_decay=epsilon_decay,
            epsilon_min=epsilon_min,
            gamma=gamma,
            lr=lr,
            replay_capacity=replay_capacity,
            batch_size=batch_size,
            target_sync_freq=target_sync_freq,
            seed=seed,
            offload_history=offload_history,
        )
        self._pso_n_particles = n_particles
        self._pso_max_iter    = max_iter
        self._pso_inertia     = inertia
        self._pso_c1          = c1
        self._pso_c2          = c2
        self._pso_rng         = np.random.default_rng(seed + 1)

    def _exhaustive_search(
        self,
        task: HealthcareTask,
        candidate_indices: List[int],
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> int:
        """
        Override parent exhaustive search with PSO inner search.

        Note: at K=3, a PSO with any reasonable particle count will
        evaluate all three candidates during random initialisation,
        making the result identical to exhaustive search by construction.
        """
        return self._pso_inner_search(task, candidate_indices, lat_bounds, eng_bounds)

    def _pso_inner_search(
        self,
        task: HealthcareTask,
        candidate_indices: List[int],
        lat_bounds: Tuple[float, float],
        eng_bounds: Tuple[float, float],
    ) -> int:
        """
        PSO restricted to the K-node candidate subspace.

        With K=3, each particle's position is an index into candidate_indices.
        The discrete optimum is the candidate with lowest cost F(j).
        """
        K = len(candidate_indices)
        if K == 1:
            return self._idx_to_node[candidate_indices[0]]

        # Evaluate exact cost for each candidate (same as exhaustive search)
        # At K=3, this is O(K) regardless of PSO overhead
        costs = {}
        for idx in candidate_indices:
            node_id = self._idx_to_node[idx]
            cost, _, _, _ = self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
            costs[node_id] = cost

        # PSO over discrete K-node space: particles maintain position as float index
        # mapped to the nearest candidate.
        # Guarantee one particle per candidate (coverage), then fill remaining randomly.
        # This ensures all K candidates are visited regardless of swarm size.
        n_particles = max(self._pso_n_particles, K)
        anchors = np.arange(K, dtype=float)                        # [0., 1., 2.]
        extras  = self._pso_rng.uniform(0, K - 1e-9,
                                         size=max(0, n_particles - K))
        positions = np.concatenate([anchors, extras])
        n_particles = len(positions)                 # actual count after concatenation
        velocities = self._pso_rng.uniform(-1, 1, size=n_particles)
        pbest_pos = positions.copy()

        def _cost_at(pos: float) -> float:
            idx = candidate_indices[int(np.clip(pos, 0, K - 1))]
            node_id = self._idx_to_node[idx]
            return costs[node_id]

        pbest_cost = np.array([_cost_at(p) for p in positions])
        gbest_idx  = int(np.argmin(pbest_cost))
        gbest_pos  = pbest_pos[gbest_idx]

        for _ in range(self._pso_max_iter):
            r1 = self._pso_rng.uniform(0, 1, size=n_particles)
            r2 = self._pso_rng.uniform(0, 1, size=n_particles)
            velocities = (
                self._pso_inertia * velocities
                + self._pso_c1 * r1 * (pbest_pos - positions)
                + self._pso_c2 * r2 * (gbest_pos - positions)
            )
            positions = np.clip(positions + velocities, 0, K - 1e-9)
            curr_costs = np.array([_cost_at(p) for p in positions])
            improved = curr_costs < pbest_cost
            pbest_pos[improved] = positions[improved]
            pbest_cost[improved] = curr_costs[improved]
            best_i = int(np.argmin(pbest_cost))
            if pbest_cost[best_i] < _cost_at(gbest_pos):
                gbest_pos = pbest_pos[best_i]

        best_candidate_idx = candidate_indices[int(np.clip(gbest_pos, 0, K - 1))]
        return self._idx_to_node[best_candidate_idx]
