from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict
from .prompts import SYSTEM_PROMPT, build_judge_prompt


class JudgeClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.provider = cfg.get("judge", {}).get("provider", "openai_compatible")
        self.model = os.getenv("JUDGE_MODEL", "qwen-plus")
        self.temperature = cfg.get("judge", {}).get("temperature", 0)
        self.max_retries = cfg.get("judge", {}).get("max_retries", 3)
        self.max_chars = cfg.get("judge", {}).get("max_chars_per_slice", 18000)
        self.client = None
        if self.provider == "openai_compatible":
            api_key = os.getenv("JUDGE_API_KEY")
            if not api_key:
                raise RuntimeError("JUDGE_API_KEY is empty. Set judge.provider=heuristic for offline demo, or fill .env for LLM Judge.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Please install openai: pip install openai") from exc
            self.client = OpenAI(api_key=api_key, base_url=os.getenv("JUDGE_BASE_URL") or None)

    def score_slice(self, issue: str, td_slice: dict, rubric: dict) -> Dict[str, Any]:
        if self.provider == "heuristic":
            return heuristic_score(td_slice, rubric)
        td_slice = dict(td_slice)
        for k in ["planning_text", "execution_text", "revision_text"]:
            if isinstance(td_slice.get(k), str):
                td_slice[k] = td_slice[k][:self.max_chars]
        prompt = build_judge_prompt(issue[:8000], td_slice, rubric)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                content = resp.choices[0].message.content or "{}"
                return validate_judge_json(json.loads(content))
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Judge failed after retries: {last_err}")


def heuristic_score(td_slice: dict, rubric: dict) -> Dict[str, Any]:
    text = "\n".join([
        td_slice.get("planning_text", ""),
        td_slice.get("execution_text", ""),
        td_slice.get("revision_text", ""),
    ]).lower()
    stages = {}
    for stage, kws in rubric.get("engineering_stages", {}).items():
        stages[stage] = any(k.lower() in text for k in kws)
    covered = sum(bool(v) for v in stages.values())
    has_structured = bool(re.search(r"(^|\n)\s*(1\.|1\)|- |\*)", td_slice.get("planning_text", "")))
    has_revision = bool(td_slice.get("revision_text", "").strip())
    if not td_slice.get("planning_text", "").strip():
        score = 0
    elif covered <= 1 and not has_structured:
        score = 1
    elif covered <= 2:
        score = 2
    elif covered == 4 and not has_revision:
        score = 3 if not has_structured else 4
    else:
        score = 5 if has_revision and covered >= 3 else min(4, covered)

    actions = re.findall(r"action=", td_slice.get("execution_text", ""))
    total = max(1, len(actions))
    aligned = min(total, covered if covered > 0 else total // 2)
    return validate_judge_json({
        "score": score,
        "classification_label": rubric.get("score_labels", {}).get(score, "Heuristic Score"),
        "reasoning": "Heuristic offline judge based on stage coverage, structured planning, and revision evidence. Use openai_compatible provider for final experiments.",
        "covered_stages": stages,
        "aligned_steps": aligned,
        "total_executed_steps": total,
        "reasonable_revisions": 1 if has_revision else 0,
        "total_revisions": 1 if has_revision else 0,
        "efficiency_score": 0.8 if covered >= 3 else 0.5,
        "evidence": [],
    })


def validate_judge_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    obj["score"] = max(0, min(5, int(obj.get("score", 0))))
    obj["classification_label"] = str(obj.get("classification_label", ""))
    obj["reasoning"] = str(obj.get("reasoning", ""))
    obj["aligned_steps"] = max(0, int(obj.get("aligned_steps", 0)))
    obj["total_executed_steps"] = max(0, int(obj.get("total_executed_steps", 0)))
    obj["reasonable_revisions"] = max(0, int(obj.get("reasonable_revisions", 0)))
    obj["total_revisions"] = max(0, int(obj.get("total_revisions", 0)))
    obj["efficiency_score"] = float(obj.get("efficiency_score", 0.0))
    obj["efficiency_score"] = max(0.0, min(1.0, obj["efficiency_score"]))
    obj.setdefault("covered_stages", {})
    obj.setdefault("evidence", [])
    return obj
