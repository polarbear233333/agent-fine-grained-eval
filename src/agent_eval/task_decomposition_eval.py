from __future__ import annotations

from typing import Any

from td_pipeline.aggregator import aggregate_scores
from td_pipeline.judge_client import heuristic_score
from td_pipeline.slice_extractor import extract_slices
from td_pipeline.trajectory_schema import Trajectory, TrajectoryStep


DEFAULT_RUBRIC = {
    "planning_triggers": {
        "strong": ["plan:", "steps:", "todo:", "I will first", "first I"],
        "revision": ["revise my plan", "new plan", "previous assumption", "instead, I should", "need to adjust"],
    },
    "engineering_stages": {
        "reproduction": ["reproduce", "run test", "pytest", "failing test"],
        "localization": ["locate", "inspect", "grep", "search", "trace"],
        "modification": ["modify", "patch", "fix", "edit", "implement"],
        "verification": ["verify", "rerun", "regression", "all tests", "passed"],
    },
    "score_labels": {
        0: "No Planning",
        1: "Weak Planning",
        2: "Partial Planning",
        3: "Standard Engineering Planning",
        4: "Systematic Planning",
        5: "Advanced Dynamic Planning",
    },
    "aggregation_weights": {"pqs": 0.4, "par": 0.3, "prq": 0.2, "ee": 0.1},
}


def trajectory_from_messages(instance_id: str, messages: list[dict[str, Any]]) -> Trajectory:
    steps: list[TrajectoryStep] = []
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "assistant")
        content = str(msg.get("content") or "")
        if role == "tool":
            steps.append(TrajectoryStep(turn_id=idx, role=role, thought="", observation=content, raw=msg))
        elif role == "assistant":
            steps.append(TrajectoryStep(turn_id=idx, role=role, thought=content, action=msg.get("tool_call") or msg.get("action"), raw=msg))
        elif role == "user":
            steps.append(TrajectoryStep(turn_id=idx, role=role, thought=content, raw=msg))
    return Trajectory(instance_id=instance_id, status="completed", steps=steps)


class TaskDecompositionEvaluator:
    def __init__(self, rubric: dict[str, Any] | None = None):
        self.rubric = rubric or DEFAULT_RUBRIC

    def evaluate_messages(self, instance_id: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if not messages:
            return {"slice_count": 0, "metrics": aggregate_scores([], self.rubric), "slices": []}
        traj = trajectory_from_messages(instance_id, messages)
        slices = extract_slices(traj, self.rubric)
        scored = []
        judge_scores = []
        for td_slice in slices:
            score = heuristic_score(td_slice.model_dump(), self.rubric)
            item = td_slice.model_dump()
            item["judge"] = score
            scored.append(item)
            judge_scores.append(score)
        return {
            "slice_count": len(slices),
            "metrics": aggregate_scores(judge_scores, self.rubric),
            "slices": scored,
        }

