from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

from .ce_schema import ContextEngineeringInstance

CE_SYSTEM_PROMPT = """You are an expert evaluator for Effective Context Engineering in autonomous coding agents.
Evaluate whether the agent preserves, compresses, discards, recalls, and applies task-critical context under a limited context budget.
Return strict JSON only."""


def build_ce_prompt(instance: ContextEngineeringInstance, agent_output: Dict[str, Any]) -> str:
    return f"""
Evaluate this Context Engineering run.

Rubric summary:
- Context Budgeting: keep/summarize/discard should prioritize Patch Causal Units (PCUs) and avoid context bloat.
- Information Retention: hard PCUs introduced early should remain available through memory/keep/summary after later noisy turns.
- Actionable Recall: retained PCUs should influence later decisions or patch semantics, not merely be copied.

Return JSON with:
{{
  "hard_pcu_recall": 0.0-1.0,
  "soft_pcu_recall": 0.0-1.0,
  "context_bloat_ratio": float,
  "memory_utilization": 0.0-1.0,
  "delayed_recall_accuracy": 0.0-1.0,
  "conflict_resolution_accuracy": 0.0-1.0,
  "noise_resistance_score": 0.0-1.0,
  "forgetting_events_count": integer,
  "actionable_recall_score": 0.0-1.0,
  "reasoning": "brief explanation",
  "evidence": ["..."]
}}

Benchmark instance:
{json.dumps(instance.model_dump(), ensure_ascii=False)[:24000]}

Agent output / context plan / trajectory summary:
{json.dumps(agent_output, ensure_ascii=False)[:24000]}
"""


class ContextJudgeClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.provider = cfg.get("judge", {}).get("provider", "heuristic")
        self.model = os.getenv("JUDGE_MODEL", "qwen-plus")
        self.temperature = cfg.get("judge", {}).get("temperature", 0)
        self.max_retries = cfg.get("judge", {}).get("max_retries", 3)
        self.client = None
        if self.provider == "openai_compatible":
            api_key = os.getenv("JUDGE_API_KEY")
            if not api_key:
                raise RuntimeError("JUDGE_API_KEY is empty. Use judge.provider=heuristic for offline demo, or fill .env for LLM Judge.")
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=os.getenv("JUDGE_BASE_URL") or None)

    def score(self, instance: ContextEngineeringInstance, agent_output: Dict[str, Any]) -> Dict[str, Any]:
        if self.provider == "heuristic":
            return heuristic_ce_score(instance, agent_output)
        prompt = build_ce_prompt(instance, agent_output)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "system", "content": CE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                return validate_ce_json(json.loads(resp.choices[0].message.content or "{}"))
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Context judge failed after retries: {last_err}")


def _flatten_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False)


def _plan_refs(agent_output: Dict[str, Any]) -> List[str]:
    plan = agent_output.get("context_plan", agent_output)
    refs: List[str] = []
    for k in ["keep", "discard"]:
        refs.extend([str(x) for x in plan.get(k, []) if x])
    for x in plan.get("summarize", []):
        if isinstance(x, dict):
            refs.append(str(x.get("ref", "")))
        else:
            refs.append(str(x))
    memory = plan.get("memory", [])
    refs.extend(_flatten_text(memory).split())
    return refs


def heuristic_ce_score(instance: ContextEngineeringInstance, agent_output: Dict[str, Any]) -> Dict[str, Any]:
    text = _flatten_text(agent_output).lower()
    refs_text = " ".join(_plan_refs(agent_output)).lower()
    retained_tokens = 0
    for s in instance.context_sources:
        if s.ref_id.lower() in refs_text or s.ref_id.lower() in text:
            retained_tokens += s.token_count
    hard = [p for p in instance.pcus if p.necessity == "hard"]
    soft = [p for p in instance.pcus if p.necessity == "soft"]

    def covered(pcu) -> bool:
        hay = text + " " + refs_text
        if pcu.pcu_id.lower() in hay:
            return True
        if any(span.ref_id.lower() in hay for span in pcu.source_spans):
            return True
        words = [w for w in re.findall(r"[a-zA-Z_]{4,}", pcu.description.lower()) if w not in {"should", "patch", "issue", "core", "test", "signal"}]
        return bool(words) and sum(w in hay for w in words[:20]) >= min(3, len(words))

    hard_recall = sum(covered(p) for p in hard) / max(1, len(hard))
    soft_recall = sum(covered(p) for p in soft) / max(1, len(soft)) if soft else 1.0
    minimal = sum(instance.context_sources[0:1][0].token_count for _ in hard) if hard and instance.context_sources else 1
    cbr = retained_tokens / max(1, minimal)
    plan = agent_output.get("context_plan", agent_output)
    mem = plan.get("memory", []) if isinstance(plan, dict) else []
    memory_util = min(1.0, len(mem) / max(1, instance.context_budget.memory_slots))
    delayed = 1.0 if hard_recall >= 0.99 and any("delayed" in t.tags for t in instance.interaction_script) else hard_recall
    conflict = 1.0 if "conflict" not in text or hard_recall >= 0.99 else 0.5
    noise = max(0.0, min(1.0, hard_recall * (1.2 if cbr <= 3 else 0.8)))
    actionable = 1.0 if any(k in text for k in ["patch", "modify", "fix", "semantic_change", "test"]) and hard_recall > 0 else 0.5 * hard_recall
    final = 0.35 * hard_recall + 0.15 * soft_recall + 0.15 * max(0.0, min(1.0, 1 / max(1.0, cbr))) + 0.15 * delayed + 0.2 * actionable
    return validate_ce_json({
        "hard_pcu_recall": hard_recall,
        "soft_pcu_recall": soft_recall,
        "context_bloat_ratio": cbr,
        "memory_utilization": memory_util,
        "delayed_recall_accuracy": delayed,
        "conflict_resolution_accuracy": conflict,
        "noise_resistance_score": noise,
        "forgetting_events_count": 0 if hard_recall >= 0.99 else 1,
        "actionable_recall_score": actionable,
        "final_ce_score": final,
        "reasoning": "Heuristic CE judge: checks whether PCU ids/ref_ids/keywords appear in keep, summarize, memory, or patch-related outputs; use LLM judge for final experiments.",
        "evidence": [],
    })


def validate_ce_json(obj: Dict[str, Any]) -> Dict[str, Any]:
    fields = ["hard_pcu_recall", "soft_pcu_recall", "memory_utilization", "delayed_recall_accuracy", "conflict_resolution_accuracy", "noise_resistance_score", "actionable_recall_score"]
    for f in fields:
        obj[f] = max(0.0, min(1.0, float(obj.get(f, 0.0))))
    obj["context_bloat_ratio"] = max(0.0, float(obj.get("context_bloat_ratio", 0.0)))
    obj["forgetting_events_count"] = max(0, int(obj.get("forgetting_events_count", 0)))
    if "final_ce_score" not in obj:
        obj["final_ce_score"] = round(0.35 * obj["hard_pcu_recall"] + 0.15 * obj["soft_pcu_recall"] + 0.15 * max(0.0, min(1.0, 1 / max(1.0, obj["context_bloat_ratio"]))) + 0.15 * obj["delayed_recall_accuracy"] + 0.2 * obj["actionable_recall_score"], 4)
    obj["final_ce_score"] = max(0.0, min(1.0, float(obj["final_ce_score"])))
    obj.setdefault("reasoning", "")
    obj.setdefault("evidence", [])
    return obj
