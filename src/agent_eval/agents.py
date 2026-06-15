from __future__ import annotations

import json
import os
import re
from typing import Any

from .schema import BenchmarkCase
from .utils import keyword_set, truncate_text


class BaseAgent:
    def run(self, case: BenchmarkCase) -> dict[str, Any]:
        raise NotImplementedError


class HeuristicContextAgent(BaseAgent):
    """Offline baseline for pipeline validation and dataset smoke runs."""

    def __init__(self, model: str = "heuristic"):
        self.model = model

    def run(self, case: BenchmarkCase) -> dict[str, Any]:
        keep: list[str] = []
        summarize: list[dict[str, str]] = []
        discard: list[str] = []
        memory: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []

        for source in case.context_sources:
            if not source.visible_to_agent:
                discard.append(source.ref_id)
                continue
            if source.source in {"issue", "tests"}:
                keep.append(source.ref_id)
            elif source.source == "logs":
                summary = self._summarize_log(source.text)
                summarize.append({"ref": source.ref_id, "summary": summary})
                if "Key failure signal:" in summary:
                    memory.append(
                        {
                            "key": f"log-{source.ref_id}",
                            "value": summary,
                            "importance": 0.65,
                        }
                    )
            elif source.source in {"discussion", "snippets"}:
                summarize.append({"ref": source.ref_id, "summary": self._summarize_noise_aware(source.text)})
            else:
                discard.append(source.ref_id)

        issue = next((s for s in case.context_sources if s.source == "issue" and s.visible_to_agent), None)
        if issue:
            for idx, fact in enumerate(self._extract_issue_facts(issue.text)[:3], start=1):
                memory.append(
                    {
                        "key": f"issue-constraint-{idx}",
                        "value": fact,
                        "importance": 0.8,
                    }
                )
        tests = [s for s in case.context_sources if s.source == "tests" and s.visible_to_agent]
        for test in tests[:2]:
            memory.append(
                {
                    "key": f"test-{test.ref_id}",
                    "value": self._summarize_noise_aware(test.text),
                    "importance": 0.75,
                }
            )

        detected_paths = case.metadata.get("detected_paths") or []
        if detected_paths and case.track in {"budgeted_patch", "long_horizon"}:
            actions.append({"type": "search", "query": " ".join(detected_paths[:3])})

        final_patch = ""
        if case.track in {"budgeted_patch", "long_horizon"}:
            observed = str(case.metadata.get("observed_final_patch") or "")
            final_patch = observed if observed.startswith("diff --git ") else "diff --git a/placeholder b/placeholder\n# no model patch generated in heuristic mode\n"

        return {
            "model": self.model,
            "context_plan": {
                "keep": keep,
                "summarize": summarize[:4],
                "discard": discard,
                "memory": memory[: case.context_budget.memory_slots],
            },
            "actions": actions,
            "final_patch": final_patch,
        }

    def _summarize_log(self, text: str) -> str:
        for line in text.splitlines():
            if re.search(r"(traceback|assertionerror|exception|error:|failed|failure|expected|actual)", line, re.I):
                return "Key failure signal: " + truncate_text(line.strip(), 300, suffix="")
        return "Diagnostic log retained as low-priority context: " + truncate_text(text.strip(), 220, suffix="")

    def _summarize_noise_aware(self, text: str) -> str:
        keys = sorted(keyword_set(text, max_words=14))
        if keys:
            return "Candidate context with possible distractors; keywords: " + ", ".join(keys)
        return truncate_text(text, 240, suffix="")

    def _extract_issue_facts(self, text: str) -> list[str]:
        facts: list[str] = []
        for raw in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = raw.strip()
            if not sentence:
                continue
            if re.search(r"\b(should|must|expected|actual|fails?|error|regression|incorrect|wrong|when|if)\b", sentence, re.I):
                facts.append(truncate_text(sentence, 320, suffix=""))
        if not facts and text.strip():
            facts.append(truncate_text(text.strip(), 320, suffix=""))
        return facts


class OpenAICompatibleContextAgent(BaseAgent):
    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY"):
        api_key = os.getenv(api_key_env) or os.getenv("JUDGE_API_KEY")
        if not api_key:
            raise RuntimeError(f"{api_key_env} or JUDGE_API_KEY is required for openai_compatible agent")
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("API_BASE")
            or os.getenv("JUDGE_BASE_URL")
            or None,
        )
        self.model = model

    def run(self, case: BenchmarkCase) -> dict[str, Any]:
        visible_sources = [s.model_dump(exclude={"text"}) | {"text": truncate_text(s.text, 2600)} for s in case.context_sources if s.visible_to_agent]
        prompt = {
            "task": "Return strict JSON for context management under budget. Include keep, summarize, discard, memory, actions, and optional final_patch.",
            "track": case.track,
            "context_budget": case.context_budget.model_dump(),
            "context_sources": visible_sources,
            "interaction_script": [t.model_dump() for t in case.interaction_script],
            "output_schema": {
                "context_plan": {
                    "keep": ["ref_id"],
                    "summarize": [{"ref": "ref_id", "summary": "short summary"}],
                    "discard": ["ref_id"],
                    "memory": [{"key": "name", "value": "remembered fact"}],
                },
                "actions": [{"type": "search|open_file|external_retrieve", "query": "..."}],
                "final_patch": "diff --git ... optional for Track B/C",
            },
        }
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a software-engineering benchmark agent. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")


def make_agent(provider: str, model: str) -> BaseAgent:
    if provider in {"heuristic", "mock", "offline"}:
        return HeuristicContextAgent(model=model)
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleContextAgent(model=model)
    raise ValueError(f"Unsupported agent provider: {provider}")
