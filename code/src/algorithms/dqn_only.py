"""
DQN-only ablation scheduler  (Fix 2 — required for any hybrid claim).

Identical to DQN-ES except the BBO local-search phase is removed.  The
DQN outputs Q-values for all candidate nodes and the scheduler greedily
selects argmax (still using epsilon-greedy exploration during training).

This baseline isolates the contribution of the BBO antenna refinement
from the deep-RL value-function learning.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.algorithms.dqn_es import DQNNetwork, ReplayBuffer
from src.config import (
    DQN_BATCH_SIZE,
    DQN_GAMMA,
    DQN_HIDDEN_DIM,
    DQN_LR,
    DQN_REPLAY_CAPACITY,
    DQN_TARGET_SYNC,
    EPSILON_DECAY,
    EPSILON_INIT,
    EPSILON_MIN,
)
from src.core.cost_function import compute_cost
from src.core.task import HealthcareTask


class DQNOnlyScheduler(BaseScheduler):
    """
    Pure DQN policy over the full candidate-node set, no BBO refinement.

    Architecture, replay buffer, target-network sync, and reward shaping
    are inherited from DQNESScheduler's design for a like-for-like
    ablation.
    """

    STATE_DIM = 6

    def __init__(
        self,
        topology,
        epsilon: float = EPSILON_INIT,
        epsilon_decay: float = EPSILON_DECAY,
        epsilon_min: float = EPSILON_MIN,
        gamma: float = DQN_GAMMA,
        lr: float = DQN_LR,
        replay_capacity: int = DQN_REPLAY_CAPACITY,
        batch_size: int = DQN_BATCH_SIZE,
        target_sync_freq: int = DQN_TARGET_SYNC,
        seed: int = 42,
        offload_history: Optional[dict] = None,
    ):
        super().__init__(topology, offload_history)

        self._idx_to_node: List[int] = self._candidate_nodes
        self._n_nodes = len(self._idx_to_node)

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_sync_freq = target_sync_freq

        self._online_net = DQNNetwork(
            state_dim=self.STATE_DIM,
            action_dim=self._n_nodes,
            hidden_dim=DQN_HIDDEN_DIM,
            seed=seed,
        )
        self._target_net = DQNNetwork(
            state_dim=self.STATE_DIM,
            action_dim=self._n_nodes,
            hidden_dim=DQN_HIDDEN_DIM,
            seed=seed,
        )
        self._target_net.copy_weights_from(self._online_net)

        self._replay = ReplayBuffer(capacity=replay_capacity, seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        self._max_rate_bps = 1e9
        self._max_load = 100
        self._max_rtt_s = 0.5

        # Trajectory log (used by Fix 7 epsilon convergence figure)
        self.epsilon_history: list[float] = []

    # ------------------------------------------------------------------
    def _state(self, task: HealthcareTask, node_id: int) -> np.ndarray:
        try:
            rate = self.topology.get_uplink_rate(task.device_id, node_id)
        except Exception:
            rate = 0.0
        node = self.topology.get_node(node_id)
        try:
            prop = self.topology.get_propagation_delay(task.device_id, node_id)
        except Exception:
            prop = 0.25
        if node.node_type == 'cloud':
            battery_norm = 1.0
        elif node.node_type == 'fog':
            battery_norm = 0.8
        elif node.node_type == 'edge':
            battery_norm = 0.6
        else:
            battery_norm = 0.3
        return np.array([
            min(rate / self._max_rate_bps, 1.0),
            min(node.current_load / self._max_load, 1.0),
            min(2.0 * prop / self._max_rtt_s, 1.0),
            battery_norm,
            float(task.ci_score),
            float(task.attack_probability),
        ], dtype=float)

    def _reward(self, task, latency_s, energy_j, privacy_risk):
        lat_bounds = (0.0, task.max_delay_s * 2.0)
        eng_bounds = (0.0, 0.1)
        cost = compute_cost(
            latency_s, energy_j, privacy_risk,
            task.ci_score, lat_bounds, eng_bounds,
        )
        phi = float(task.ci_score)
        if latency_s <= task.max_delay_s:
            sla_term = phi * 0.5
        else:
            v = (latency_s - task.max_delay_s) / max(task.max_delay_s, 1e-6)
            sla_term = -phi * min(v, 2.0)
        return float(-cost + sla_term)

    # ------------------------------------------------------------------
    def select_node(self, task: HealthcareTask) -> int:
        if self._n_nodes == 1:
            nid = self._idx_to_node[0]
            self.record_decision(task.device_id, nid)
            return nid

        # Bounds for outcome computation
        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)

        # State based on the representative candidate node
        state = self._state(task, self._idx_to_node[0])

        # Epsilon-greedy argmax over full candidate set
        if self._rng.random() < self.epsilon:
            action_idx = int(self._rng.integers(0, self._n_nodes))
        else:
            q = self._online_net.forward(state)
            action_idx = int(np.argmax(q))

        node_id = self._idx_to_node[action_idx]

        # Outcome for transition logging
        _, latency_s, energy_j, privacy_risk = self.evaluate_node(
            task, node_id, lat_bounds, eng_bounds,
        )
        reward = self._reward(task, latency_s, energy_j, privacy_risk)
        next_state = self._state(task, node_id)
        self._replay.push(state, action_idx, reward, next_state, False)

        # Online policy update
        if len(self._replay) >= self.batch_size:
            self._update_policy()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        self._step_count += 1
        if self._step_count % self.target_sync_freq == 0:
            self._target_net.copy_weights_from(self._online_net)

        self.record_decision(task.device_id, node_id)
        return node_id

    def _update_policy(self) -> float:
        states, actions, rewards, next_states, dones = self._replay.sample(
            self.batch_size,
        )
        q_current = self._online_net.forward(states)
        q_next = self._target_net.forward(next_states).max(axis=1)
        targets = q_current.copy()
        for i in range(self.batch_size):
            td = rewards[i] + (1.0 - dones[i]) * self.gamma * q_next[i]
            targets[i, actions[i]] = td
        return self._online_net.update(states, targets, lr=self.lr)
