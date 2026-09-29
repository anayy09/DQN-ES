"""
Static tier rule (plan E2; self-review F3).

ECG analysis tasks run on the edge gateway; every other task type runs
locally on the wearable.  No state, no learning, no cost evaluation.
"""

from __future__ import annotations

from typing import Optional

from src.algorithms.base_scheduler import BaseScheduler
from src.core.task import HealthcareTask

EDGE_TASK_TYPES = frozenset({'ecg_analysis'})


class StaticTierScheduler(BaseScheduler):
    """ECG -> edge gateway, all other task types -> local execution."""

    def __init__(self, topology, offload_history: Optional[dict] = None):
        super().__init__(topology, offload_history)
        edges = [nid for nid, n in topology.nodes.items() if n.node_type == 'edge']
        if not edges:
            raise RuntimeError('StaticTierScheduler requires an edge node.')
        self._edge_id = min(edges)

    def select_node(self, task: HealthcareTask) -> int:
        if task.task_type == 'unknown':
            raise ValueError('StaticTierScheduler needs task.task_type set.')
        node_id = (self._edge_id if task.task_type in EDGE_TASK_TYPES
                   else task.device_id)
        self.record_decision(task.device_id, node_id)
        return node_id
