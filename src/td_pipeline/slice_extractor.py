from __future__ import annotations

import re
from .trajectory_schema import Trajectory, TaskDecompositionSlice


def _text(step) -> str:
    action = step.action
    if isinstance(action, dict):
        action_text = " ".join(str(v) for v in action.values())
    else:
        action_text = str(action or "")
    return "\n".join([step.thought or "", action_text, step.observation or ""])


def contains_any(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def looks_like_numbered_plan(text: str) -> bool:
    return bool(re.search(r"(^|\n)\s*(1\.|1\)|- |\*)\s+", text)) and any(
        w in text.lower() for w in ["plan", "step", "first", "reproduce", "locate", "verify", "todo"]
    )


def is_planning_trigger(text: str, rubric: dict) -> bool:
    strong = rubric.get("planning_triggers", {}).get("strong", [])
    return contains_any(text, strong) or looks_like_numbered_plan(text)


def is_revision(text: str, rubric: dict) -> bool:
    rev = rubric.get("planning_triggers", {}).get("revision", [])
    return contains_any(text, rev)


def is_completion(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in ["all tests pass", "passed", "verify", "verification", "final", "done", "completed"])


def extract_slices(traj: Trajectory, rubric: dict, max_window: int = 30) -> list[TaskDecompositionSlice]:
    slices = []
    i = 0
    while i < len(traj.steps):
        step = traj.steps[i]
        combined = _text(step)
        if not is_planning_trigger(combined, rubric):
            i += 1
            continue

        start = i
        end = min(len(traj.steps) - 1, i + max_window)
        revision_texts = []
        completion_reason = "max_window"
        for j in range(i + 1, min(len(traj.steps), i + max_window + 1)):
            t = _text(traj.steps[j])
            if is_revision(t, rubric):
                revision_texts.append(t[:1500])
            if j > i + 2 and is_completion(t):
                end = j
                completion_reason = "verification_or_completion_detected"
                break
            if j > i + 5 and is_planning_trigger(t, rubric):
                end = j - 1
                completion_reason = "next_planning_trigger"
                break

        execution_chunks = []
        for s in traj.steps[start + 1:end + 1]:
            execution_chunks.append(f"[turn {s.turn_id}] thought={s.thought}\naction={s.action}\nobservation={s.observation}")

        slices.append(TaskDecompositionSlice(
            task_decomposition_id=len(slices) + 1,
            start_turn=traj.steps[start].turn_id,
            end_turn=traj.steps[end].turn_id,
            planning_text=traj.steps[start].thought or combined,
            execution_text="\n\n".join(execution_chunks),
            revision_text="\n\n".join(revision_texts),
            completion_reason=completion_reason,
            step_ids=[s.turn_id for s in traj.steps[start:end + 1]],
        ))
        i = end + 1
    return slices
