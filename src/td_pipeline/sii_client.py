from __future__ import annotations

import os
from typing import Any, Dict
import requests
from .trajectory_schema import Case, Trajectory, TrajectoryStep


class SIIClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mode = cfg.get("sii", {}).get("mode", "http")
        self.base_url = os.getenv("SII_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("SII_API_KEY", "")
        self.endpoint = cfg.get("sii", {}).get("endpoint", "/run")
        self.timeout = cfg.get("sii", {}).get("timeout_sec", 600)

    def run_case(self, case: Case) -> Trajectory:
        if self.mode == "mock":
            return self._mock_run(case)
        payload = case.model_dump()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}{self.endpoint}"
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return normalize_sii_response(resp.json(), case.instance_id)

    def _mock_run(self, case: Case) -> Trajectory:
        steps = [
            TrajectoryStep(turn_id=0, thought="Plan:\n1. Reproduce the reported issue with a focused test.\n2. Locate the related function using search.\n3. Modify the implementation.\n4. Run the failing test and related regression tests to verify.", action=None),
            TrajectoryStep(turn_id=1, thought="I will first reproduce the issue.", action={"type":"shell", "command":"pytest tests/test_bug.py"}, observation="1 failed"),
            TrajectoryStep(turn_id=2, thought="Now I need to locate the relevant code path.", action={"type":"shell", "command":"grep -R \"target_function\" -n src tests"}, observation="src/pkg/core.py:10"),
            TrajectoryStep(turn_id=3, thought="The previous assumption seems correct; I will patch the implementation.", action={"type":"edit", "file":"src/pkg/core.py"}, observation="patched"),
            TrajectoryStep(turn_id=4, thought="Now verify the fix and run related tests.", action={"type":"shell", "command":"pytest tests/test_bug.py tests/test_core.py"}, observation="2 passed"),
        ]
        return Trajectory(instance_id=case.instance_id, status="completed", steps=steps)


def normalize_sii_response(data: Dict[str, Any], fallback_instance_id: str) -> Trajectory:
    raw_steps = data.get("trajectory") or data.get("steps") or data.get("messages") or []
    steps = []
    for i, item in enumerate(raw_steps):
        if isinstance(item, str):
            steps.append(TrajectoryStep(turn_id=i, thought=item, raw={"text": item}))
            continue
        thought = item.get("thought") or item.get("reasoning") or item.get("content") or ""
        action = item.get("action") or item.get("tool_call") or item.get("command")
        obs = item.get("observation") or item.get("result") or item.get("tool_result") or ""
        steps.append(TrajectoryStep(
            turn_id=int(item.get("turn_id", i)),
            role=item.get("role", "assistant"),
            thought=thought,
            action=action,
            observation=str(obs) if obs is not None else "",
            timestamp=item.get("timestamp"),
            raw=item,
        ))
    return Trajectory(
        instance_id=data.get("instance_id", fallback_instance_id),
        status=data.get("status", "unknown"),
        steps=steps,
        raw=data,
    )
