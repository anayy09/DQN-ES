"""
DQN-only ablation scheduler.

The DQN-ES learner with the enumeration step removed: the executed node is
argmax_a Q(s, a) over the network destinations, with epsilon-greedy
exploration (with probability epsilon a uniformly random destination).

It is DQNESScheduler with the candidate set fixed to K = 1, so it shares the
DQN-ES state vector (CI, p_atk and per-node features for every destination),
reward, transition storage, network size and hyperparameters.  A top-1
candidate set makes the enumeration step a no-op, which is exactly the
ablation.
"""

from __future__ import annotations

from typing import Optional

from src.algorithms.dqn_es import DQNESScheduler
from src.config import (
    DQN_BATCH_SIZE,
    DQN_GAMMA,
    DQN_LR,
    DQN_REPLAY_CAPACITY,
    DQN_TARGET_SYNC,
    EPSILON_DECAY,
    EPSILON_INIT,
    EPSILON_MIN,
    GLOBAL_SEED,
)


class DQNOnlyScheduler(DQNESScheduler):
    """Greedy (epsilon-greedy) DQN policy over all destinations; K = 1."""

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
        seed: int = GLOBAL_SEED,
        offload_history: Optional[dict] = None,
    ):
        super().__init__(
            topology,
            n_candidate_nodes=1,
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
