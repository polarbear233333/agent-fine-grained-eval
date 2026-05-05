from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Case(BaseModel):
    instance_id: str
    repo: Optional[str] = None
    base_commit: Optional[str] = None
    problem_statement: str
    hints_text: Optional[str] = ""
    patch: Optional[str] = None
    test_patch: Optional[str] = None


class TrajectoryStep(BaseModel):
    turn_id: int
    role: str = "assistant"
    thought: Optional[str] = ""
    action: Optional[Dict[str, Any] | str] = None
    observation: Optional[str] = ""
    timestamp: Optional[str] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


class Trajectory(BaseModel):
    instance_id: str
    status: str = "unknown"
    steps: List[TrajectoryStep]
    raw: Dict[str, Any] = Field(default_factory=dict)


class TaskDecompositionSlice(BaseModel):
    task_decomposition_id: int
    start_turn: int
    end_turn: int
    planning_text: str
    execution_text: str
    revision_text: str = ""
    completion_reason: str = ""
    step_ids: List[int]
