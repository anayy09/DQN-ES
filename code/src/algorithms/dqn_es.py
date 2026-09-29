"""
Hybrid DQN-ES Scheduler — core research contribution.

Architecture:
  1. DQN (numpy-based, no PyTorch dependency):
     - State: Global task properties (CI, attack prob) + node-specific features (bw, cpu, rtt, battery) for all M nodes.
     - Action: selects top-K candidate node indices
     - Reward: r = -F(x) + Φ·penalty_SLA  (CI-modulated)

  2. Exhaustive Search (ES):
     - Refines the DQN-selected K-node subspace by exhaustively evaluating the exact cost function F(x) for each candidate.
     - Selects the absolute best candidate within the subspace.

  3. Hybrid flow:
     - DQN narrows search to K ≪ N candidate nodes
     - ES performs fine-grained combinatorial search within that subspace
     - DQN policy is updated online via experience replay (DQN training).
       A transition (s_t, a_t, r_t, s_{t+1}) is stored when decision t+1
       arrives, so s_{t+1} is the state at the next scheduling decision.
       a_t is the executed action (argmin F within the top-K set); Q-learning
       is off-policy, so any behaviour policy with coverage is valid.

References:
  Mnih, V. et al. (2015). Human-level control through deep reinforcement
  learning. Nature, 518(7540), 529–533.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.algorithms.base_scheduler import BaseScheduler
from src.config import (
    DQN_TOP_K,
    DQN_BATCH_SIZE,
    DQN_GAMMA,
    DQN_HIDDEN_DIM,
    DQN_LR,
    DQN_REPLAY_CAPACITY,
    DQN_TARGET_SYNC,
    EPSILON_DECAY,
    EPSILON_INIT,
    EPSILON_MIN,
    GLOBAL_SEED,
)
from src.core.cost_function import (
    compute_cost,
    compute_normalized_weights,
    compute_offload_energy,
    compute_offload_latency,
    estimate_bounds,
)
from src.core.task import HealthcareTask


# ===========================================================================
# Experience Replay Buffer
# ===========================================================================

class ReplayBuffer:
    """
    Fixed-capacity circular experience replay buffer for DQN training.
    """
    def __init__(self, capacity: int = 10_000, seed: Optional[int] = None):
        self.buffer: deque = deque(maxlen=capacity)
        # Own RNG so minibatch sampling depends only on the scheduler seed,
        # not on the process-global `random` state.
        self._rng = random.Random(seed)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.buffer.append((state.copy(), int(action), float(reward), next_state.copy(), bool(done)))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = self._rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=float),
            np.array(actions, dtype=int),
            np.array(rewards, dtype=float),
            np.array(next_states, dtype=float),
            np.array(dones, dtype=float),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ===========================================================================
# Lightweight Numpy-based DQN Network
# ===========================================================================

class DQNNetwork:
    """
    Two-hidden-layer fully connected network implemented in pure NumPy.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)

        def _xavier(fan_in: int, fan_out: int) -> np.ndarray:
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self.W1 = _xavier(state_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = _xavier(hidden_dim, hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        self.W3 = _xavier(hidden_dim, action_dim)
        self.b3 = np.zeros(action_dim)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        h1 = self._relu(x @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        q = h2 @ self.W3 + self.b3

        return q[0] if single else q

    def update(self, states: np.ndarray, targets: np.ndarray, lr: float = 0.001) -> float:
        batch = states.shape[0]

        z1 = states @ self.W1 + self.b1
        h1 = self._relu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = self._relu(z2)
        q  = h2 @ self.W3 + self.b3

        diff = q - targets
        loss = float(np.mean(diff ** 2))

        dq = 2.0 * diff / batch

        dW3 = h2.T @ dq
        db3 = dq.sum(axis=0)
        dh2 = dq @ self.W3.T

        dz2 = dh2 * (z2 > 0).astype(float)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T

        dz1 = dh1 * (z1 > 0).astype(float)
        dW1 = states.T @ dz1
        db1 = dz1.sum(axis=0)

        def _clip_and_apply(param: np.ndarray, grad: np.ndarray) -> np.ndarray:
            norm = float(np.linalg.norm(grad))
            if norm > 5.0:
                grad = grad * (5.0 / norm)
            return param - lr * grad

        self.W1 = _clip_and_apply(self.W1, dW1)
        self.b1 = _clip_and_apply(self.b1, db1)
        self.W2 = _clip_and_apply(self.W2, dW2)
        self.b2 = _clip_and_apply(self.b2, db2)
        self.W3 = _clip_and_apply(self.W3, dW3)
        self.b3 = _clip_and_apply(self.b3, db3)

        return loss

    def copy_weights_from(self, other: 'DQNNetwork') -> None:
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


# ===========================================================================
# Hybrid DQN-ES Scheduler
# ===========================================================================

class DQNESScheduler(BaseScheduler):
    """
    Hybrid DQN-ES task offloading scheduler — core research contribution.

    DQN compresses the search space to top-K candidate nodes.
    Exhaustive Search (ES) evaluates exact cost F(x) within that K-node subspace.
    CI modulates the reward signal to prioritise critical tasks.
    """

    def __init__(
        self,
        topology,
        n_candidate_nodes: int = DQN_TOP_K,
        epsilon: float = EPSILON_INIT,
        epsilon_decay: float = EPSILON_DECAY,
        epsilon_min: float = EPSILON_MIN,
        gamma: float = DQN_GAMMA,
        lr: float = DQN_LR,
        replay_capacity: int = DQN_REPLAY_CAPACITY,
        batch_size: int = DQN_BATCH_SIZE,
        target_sync_freq: int = DQN_TARGET_SYNC,
        seed: int = GLOBAL_SEED,
        offload_history: Optional[dict] = None,
    ):
        super().__init__(topology, offload_history)

        # K: size of the DQN candidate set over the network destinations
        # (edge, fog nodes, cloud; local execution is not an action).
        if not 1 <= n_candidate_nodes <= len(self._candidate_nodes):
            raise ValueError(
                f'K must be in 1..{len(self._candidate_nodes)}, '
                f'got {n_candidate_nodes}')
        self.n_candidate_nodes = int(n_candidate_nodes)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_sync_freq = target_sync_freq

        self._n_nodes = len(self._candidate_nodes)
        self._idx_to_node: List[int] = self._candidate_nodes
        
        # State dimension: CI (1) + Attack Prob (1) + 4 features per node
        self.state_dim = 2 + 4 * self._n_nodes

        self._online_net = DQNNetwork(
            state_dim=self.state_dim,
            action_dim=self._n_nodes,
            hidden_dim=DQN_HIDDEN_DIM,
            seed=seed,
        )
        self._target_net = DQNNetwork(
            state_dim=self.state_dim,
            action_dim=self._n_nodes,
            hidden_dim=DQN_HIDDEN_DIM,
            seed=seed,
        )
        self._target_net.copy_weights_from(self._online_net)

        self._replay = ReplayBuffer(capacity=replay_capacity, seed=seed)
        # (state, action_idx, reward) of the last decision, awaiting s_{t+1}
        self._pending: Optional[Tuple[np.ndarray, int, float]] = None

        self._step_count = 0
        self._total_loss = 0.0
        self._rng = np.random.default_rng(seed)

        self._max_rate_bps = 1e9          # 1 Gbps reference
        # Load feature = tasks at the node / 10, capped at 1 (D16: occupancy
        # is now held until completion, so this feature varies).
        self._max_load = 10
        self._max_rtt_s = 0.5            # 500 ms reference RTT

        self.epsilon_history: List[float] = []
        self.dispatch_times_ms: List[float] = []
        # Per-decision diagnostics read by the environment logger (plan E11)
        self.last_decision_info: dict = {}
        self._last_q: Optional[np.ndarray] = None
        self._last_explored: bool = False
        self._last_forward_s: float = 0.0

    def get_state(self, task: HealthcareTask) -> np.ndarray:
        """
        Build the expanded state vector including node-specific queue statistics.
        [ci, attack_prob, node0_bw, node0_cpu, node0_rtt, node0_bat, node1_bw, ...]
        """
        state_features = [float(task.ci_score), float(task.attack_probability)]
        
        for node_id in self._idx_to_node:
            try:
                rate = self.topology.get_uplink_rate(task.device_id, node_id)
            except Exception:
                rate = 0.0
            bw_norm = min(rate / self._max_rate_bps, 1.0)

            node = self.topology.get_node(node_id)
            cpu_util = min(node.current_load / self._max_load, 1.0)

            try:
                prop = self.topology.get_propagation_delay(task.device_id, node_id)
            except Exception:
                prop = 0.25
            rtt_norm = min(2.0 * prop / self._max_rtt_s, 1.0)

            node_type = node.node_type
            if node_type == 'cloud':
                battery_norm = 1.0
            elif node_type == 'fog':
                battery_norm = 0.8
            elif node_type == 'edge':
                battery_norm = 0.6
            else:
                battery_norm = 0.3
                
            state_features.extend([bw_norm, cpu_util, rtt_norm, battery_norm])

        return np.array(state_features, dtype=float)

    def compute_reward(self, task: HealthcareTask, node_id: int, latency_s: float, energy_j: float, privacy_risk: float) -> float:
        lat_bounds = (0.0, task.max_delay_s * 2.0)
        eng_bounds = (0.0, 0.1)

        cost = compute_cost(
            latency_s, energy_j, privacy_risk,
            task.ci_score, lat_bounds, eng_bounds,
        )

        sla_met = latency_s <= task.max_delay_s
        phi = float(task.ci_score)

        if sla_met:
            sla_term = phi * 0.5
        else:
            violation_ratio = (latency_s - task.max_delay_s) / max(task.max_delay_s, 1e-6)
            sla_term = -phi * min(violation_ratio, 2.0)

        return float(-cost + sla_term)

    def select_node(self, task: HealthcareTask) -> int:
        if self._n_nodes == 1:
            node_id = self._idx_to_node[0]
            self.record_decision(task.device_id, node_id)
            return node_id

        _t = time.perf_counter
        t0 = _t()
        lat_bounds, eng_bounds = self.estimate_feasible_bounds(task)
        t_bounds = _t() - t0

        # Expanded state representation
        state = self.get_state(task)

        # Complete the previous decision's transition: its successor state
        # s_{t+1} is the state observed at this (the next) scheduling
        # decision.  The episode is one continuing stream, so done = False.
        t0 = _t()
        td_loss = float('nan')
        if self._pending is not None:
            p_state, p_action, p_reward = self._pending
            self._replay.push(p_state, p_action, p_reward, state, False)
            self._pending = None
            if len(self._replay) >= self.batch_size:
                td_loss = self.update_policy(self.batch_size)
        t_update = _t() - t0

        eps_used = self.epsilon
        t_dispatch_start = _t()
        top_k_indices = self._dqn_select_top_k(state)
        t_enum0 = _t()

        # Exhaustive search within Top-K subspace
        best_node_id = self._exhaustive_search(task, top_k_indices, lat_bounds, eng_bounds)
        t_end = _t()
        self.dispatch_times_ms.append((t_end - t_dispatch_start) * 1000.0)

        # Diagnostics (outside the timed region): Q ordering of the executed
        # action.  On exploratory steps Q is evaluated here only for logging;
        # it does not touch any RNG, so decisions are unchanged.
        q_values = (self._last_q if self._last_q is not None
                    else self._online_net.forward(state))
        order = np.argsort(q_values)[::-1]
        exec_idx = self._idx_to_node.index(best_node_id)
        self.last_decision_info = {
            'explored': bool(self._last_explored),
            'epsilon': float(eps_used),
            'q_argmax_node': int(self._idx_to_node[int(order[0])]),
            'exec_q_rank': int(np.where(order == exec_idx)[0][0]),
            'candidates': [self._idx_to_node[i] for i in top_k_indices],
            # Decision-time split (plan E4).  forward = Q forward pass on the
            # decision path (0 on exploratory steps); enum = inner
            # enumeration over the K candidates; update = replay push +
            # minibatch update, which is off the decision path.
            't_bounds_ms': t_bounds * 1000.0,
            't_forward_ms': self._last_forward_s * 1000.0,
            't_enum_ms': (t_end - t_enum0) * 1000.0,
            't_update_ms': t_update * 1000.0,
            # E11: loss of the minibatch update made at this decision (NaN if
            # none) and max Q of the online network at this state
            'td_loss': float(td_loss),
            'q_max': float(np.max(q_values)),
        }

        _, latency_s, energy_j, privacy_risk = self.evaluate_node(task, best_node_id, lat_bounds, eng_bounds)

        action_idx = self._idx_to_node.index(best_node_id)
        reward = self.compute_reward(task, best_node_id, latency_s, energy_j, privacy_risk)
        # Stored when the next decision arrives (see above).
        self._pending = (state, action_idx, reward)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

        self._step_count += 1

        if self._step_count % self.target_sync_freq == 0:
            self._target_net.copy_weights_from(self._online_net)

        self.record_decision(task.device_id, best_node_id)
        return best_node_id

    def _dqn_select_top_k(self, state: np.ndarray) -> List[int]:
        K = self.n_candidate_nodes

        if self._rng.random() < self.epsilon:
            # Exploration: a uniformly random K-subset of the destinations
            indices = self._rng.choice(self._n_nodes, size=min(K, self._n_nodes), replace=False).tolist()
            self._last_q, self._last_explored = None, True
            self._last_forward_s = 0.0
        else:
            t0 = time.perf_counter()
            q_values = self._online_net.forward(state)
            top_k_idx = np.argsort(q_values)[::-1][:K]
            indices = top_k_idx.tolist()
            self._last_forward_s = time.perf_counter() - t0
            self._last_q, self._last_explored = q_values, False

        return indices

    def _exhaustive_search(self, task: HealthcareTask, candidate_indices: List[int], lat_bounds: Tuple[float, float], eng_bounds: Tuple[float, float]) -> int:
        best_node = -1
        best_cost = float('inf')
        
        for idx in candidate_indices:
            node_id = self._idx_to_node[idx]
            cost, _, _, _ = self.evaluate_node(task, node_id, lat_bounds, eng_bounds)
            if cost < best_cost:
                best_cost = cost
                best_node = node_id
                
        return best_node

    def update_policy(self, batch_size: int = 32) -> float:
        if len(self._replay) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self._replay.sample(batch_size)

        q_current = self._online_net.forward(states)
        q_next = self._target_net.forward(next_states)
        max_q_next = q_next.max(axis=1)

        targets = q_current.copy()
        for i in range(batch_size):
            td_target = rewards[i] + (1.0 - dones[i]) * self.gamma * max_q_next[i]
            targets[i, actions[i]] = td_target

        loss = self._online_net.update(states, targets, lr=self.lr)
        self._total_loss += loss
        return loss

    def get_diagnostics(self) -> dict:
        return {
            'epsilon': self.epsilon,
            'replay_size': len(self._replay),
            'step_count': self._step_count,
            'avg_loss': (self._total_loss / max(self._step_count, 1)),
        }
