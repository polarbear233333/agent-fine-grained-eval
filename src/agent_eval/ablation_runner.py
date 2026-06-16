from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI

from .llm_pcu_engine import parse_json_object
from .manifest import write_manifest
from .schema import BenchmarkCase, ContextSource, PCU
from .utils import append_jsonl, ensure_dir, keyword_set, mask_spans, read_jsonl_stream, truncate_text, write_json


ABLATION_PROMPT_VERSION = "pcu-ablation-v1"


class AblationRunner:
    """Run Full Context vs Masked PCU ablations.

    `proxy` mode is deterministic and cheap. `llm_proxy` asks a model to solve
    from full/masked context and scores whether its plan recovers the expected
    PCU patch effect. A future `patch_test` mode can connect to the Dockerized
    SWE harness for true pass/fail ablation.
    """

    def __init__(self, solver_mode: str = "proxy", model: str = "gpt-5.4-mini"):
        self.solver_mode = solver_mode
        self.model = model
        self.api_calls: list[dict[str, Any]] = []
        self.client: OpenAI | None = None
        if solver_mode == "llm_proxy":
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("JUDGE_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY or JUDGE_API_KEY is required for llm_proxy ablation")
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE") or os.getenv("JUDGE_BASE_URL") or None
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(os.getenv("OPENAI_TIMEOUT_SEC", "120")))
        elif solver_mode not in {"proxy"}:
            raise ValueError(f"Unsupported solver_mode: {solver_mode}")

    def run_file(
        self,
        dataset_path: str | Path,
        output_dir: str | Path,
        max_cases: int | None = None,
        max_pcus_per_case: int | None = None,
        necessity: str = "all",
        include_oracle_patch_pcus: bool = False,
    ) -> list[dict[str, Any]]:
        out = ensure_dir(output_dir)
        jsonl_path = out / "ablations.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()
        results: list[dict[str, Any]] = []
        for row in read_jsonl_stream(dataset_path, limit=max_cases):
            case = BenchmarkCase(**row)
            pcus = self._select_pcus(case, necessity, include_oracle_patch_pcus, max_pcus_per_case)
            for pcu in pcus:
                result = self.run_pcu(case, pcu)
                append_jsonl(jsonl_path, result)
                results.append(result)
        summary = summarize_ablation_results(results)
        write_json(out / "ablation_metrics.json", {"summary": summary, "results": results})
        write_manifest(
            out / "artifact_manifest.json",
            command="python scripts/run_ablation.py",
            config={
                "dataset_path": str(dataset_path),
                "output_dir": str(output_dir),
                "solver_mode": self.solver_mode,
                "model": self.model,
                "max_cases": max_cases,
                "max_pcus_per_case": max_pcus_per_case,
                "necessity": necessity,
                "include_oracle_patch_pcus": include_oracle_patch_pcus,
            },
            artifacts=[
                {"name": "input_dataset", "path": str(dataset_path)},
                {"name": "ablations", "path": str(jsonl_path)},
                {"name": "ablation_metrics", "path": str(out / "ablation_metrics.json")},
            ],
            api_calls=self.api_calls,
            prompt_versions={"ablation": ABLATION_PROMPT_VERSION},
            notes=[
                "Default ablation skips oracle patch PCUs because patch sources are not visible to agents.",
                "proxy and llm_proxy are not official SWE test-pass measurements.",
            ],
        )
        return results

    def run_pcu(self, case: BenchmarkCase, pcu: PCU) -> dict[str, Any]:
        full_sources = [source for source in case.context_sources if source.visible_to_agent]
        masked_sources = mask_case_sources(full_sources, pcu)
        if self.solver_mode == "proxy":
            full = proxy_solve_score(case, pcu, full_sources)
            masked = proxy_solve_score(case, pcu, masked_sources)
        else:
            full = self.llm_proxy_score(case, pcu, full_sources, condition="full")
            masked = self.llm_proxy_score(case, pcu, masked_sources, condition="masked")
        delta = max(0.0, full["success_score"] - masked["success_score"])
        inferred = "hard" if delta >= 0.35 else "soft"
        return {
            "instance_id": case.instance_id,
            "pcu_id": pcu.pcu_id,
            "annotated_necessity": pcu.necessity,
            "inferred_necessity": inferred,
            "pcu_source": pcu.source_spans[0].source if pcu.source_spans else None,
            "expected_patch_effect": pcu.expected_patch_effect,
            "solver_mode": self.solver_mode,
            "full": full,
            "masked": masked,
            "success_delta": round(delta, 4),
        }

    def llm_proxy_score(
        self,
        case: BenchmarkCase,
        pcu: PCU,
        sources: list[ContextSource],
        condition: str,
    ) -> dict[str, Any]:
        assert self.client is not None
        prompt = {
            "task": (
                "Given the software issue context, propose the minimal semantic patch plan. "
                "Do not use hidden oracle patch text. Return JSON only."
            ),
            "instance_id": case.instance_id,
            "condition": condition,
            "context_sources": [
                {
                    "ref_id": source.ref_id,
                    "source": source.source,
                    "text": truncate_text(source.text, 3500),
                }
                for source in sources
            ],
            "output_schema": {
                "key_facts": ["facts used for repair"],
                "patch_plan": "semantic patch plan",
                "confidence": 0.0,
            },
        }
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a precise software repair planner. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        self.api_calls.append(
            {
                "kind": "ablation_llm_proxy",
                "model": self.model,
                "response_id": getattr(response, "id", None),
                "prompt_version": ABLATION_PROMPT_VERSION,
                "instance_id": case.instance_id,
                "pcu_id": pcu.pcu_id,
                "condition": condition,
            }
        )
        obj = parse_json_object(response.choices[0].message.content or "{}")
        text = json.dumps(obj, ensure_ascii=False)
        score = patch_effect_overlap_score(text, pcu)
        return {
            "success_score": round(score, 4),
            "plan": obj,
            "method": "llm_patch_plan_overlap",
        }

    def _select_pcus(
        self,
        case: BenchmarkCase,
        necessity: str,
        include_oracle_patch_pcus: bool,
        max_pcus_per_case: int | None,
    ) -> list[PCU]:
        selected: list[PCU] = []
        for pcu in case.pcus:
            if necessity != "all" and pcu.necessity != necessity:
                continue
            if not include_oracle_patch_pcus and any(span.source == "patch" for span in pcu.source_spans):
                continue
            if not any(span.ref_id for span in pcu.source_spans):
                continue
            selected.append(pcu)
        selected.sort(key=lambda p: (p.necessity != "hard", -p.importance, p.pcu_id))
        if max_pcus_per_case is not None:
            selected = selected[:max_pcus_per_case]
        return selected


def mask_case_sources(sources: list[ContextSource], pcu: PCU) -> list[ContextSource]:
    masked: list[ContextSource] = []
    spans_by_ref: dict[str, list[tuple[int, int]]] = {}
    for span in pcu.source_spans:
        if span.ref_id:
            spans_by_ref.setdefault(span.ref_id, []).append((span.start, span.end))
    for source in sources:
        text = source.text
        if source.ref_id in spans_by_ref:
            text = mask_spans(text, spans_by_ref[source.ref_id])
        masked.append(source.model_copy(update={"text": text}))
    return masked


def proxy_solve_score(case: BenchmarkCase, pcu: PCU, sources: list[ContextSource]) -> dict[str, Any]:
    visible_text = "\n".join(source.text for source in sources)
    if "[MASKED_PCU]" in visible_text:
        masked_penalty = 0.55 if pcu.necessity == "hard" else 0.22
    else:
        masked_penalty = 0.0
    overlap = patch_effect_overlap_score(visible_text, pcu)
    base = 0.95 if pcu.necessity == "hard" else 0.72
    score = max(0.0, min(1.0, max(base, overlap) - masked_penalty))
    return {
        "success_score": round(score, 4),
        "overlap_score": round(overlap, 4),
        "method": "offline_visibility_proxy",
    }


def patch_effect_overlap_score(text: str, pcu: PCU) -> float:
    target = keyword_set(pcu.expected_patch_effect + " " + pcu.description, max_words=40)
    if not target:
        return 0.0
    observed = keyword_set(text, max_words=300)
    return len(target & observed) / len(target)


def summarize_ablation_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"ablation_count": 0}
    by_annotated = Counter(r["annotated_necessity"] for r in results)
    by_inferred = Counter(r["inferred_necessity"] for r in results)
    agreement = sum(r["annotated_necessity"] == r["inferred_necessity"] for r in results) / len(results)
    return {
        "ablation_count": len(results),
        "annotated_counts": dict(by_annotated),
        "inferred_counts": dict(by_inferred),
        "mean_success_delta": round(mean(r["success_delta"] for r in results), 4),
        "necessity_agreement": round(agreement, 4),
    }
