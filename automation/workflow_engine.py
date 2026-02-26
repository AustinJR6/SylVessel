from __future__ import annotations

from typing import Any, Dict, List

from .execution_state import ExecutionState
from .task_graph import TaskNode


class WorkflowEngine:
    def __init__(self):
        self._runs: Dict[str, ExecutionState] = {}

    def start(self, workflow_id: str, tasks: List[TaskNode]) -> ExecutionState:
        state = ExecutionState(workflow_id=workflow_id, status="running", context={"tasks": [t.__dict__ for t in tasks]})
        self._runs[workflow_id] = state
        return state

    def update(self, workflow_id: str, status: str, patch: Dict[str, Any] | None = None) -> ExecutionState:
        state = self._runs.setdefault(workflow_id, ExecutionState(workflow_id=workflow_id))
        state.status = status
        if patch:
            state.context.update(patch)
        return state

    def get(self, workflow_id: str) -> ExecutionState | None:
        return self._runs.get(workflow_id)
