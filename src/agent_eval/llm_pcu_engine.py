from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any

from openai import OpenAI

from .pcu_engine import PCUEngine
from .schema import AblationEvidence, ContextSource, PCU, SourceSpan
from .utils import stable_id, truncate_text


PCU_PROMPT_VERSION = "llm-pcu-v1"

PCU_SYSTEM_PROMPT = """You are a senior software-engineering benchmark annotator.
Your job is to identify Patch Causal Units (PCUs) for one SWE-bench-style case.

A PCU is a minimal causal information unit: if a software agent does not know or use it, it is unlikely to generate the correct patch.

Return valid JSON only. Do not use markdown.
"""


class LLMPCUEngine:
    """LLM-assisted PCU extractor with deterministic span grounding.

    The model decides which facts are causally important for this particular
    case. The script then grounds model-provided evidence quotes back to exact
    character offsets in `context_sources`. This keeps PCU construction
    case-specific without trusting the model to invent offsets.
    """

    def __init__(
        self,
        model: str,
        provider: str = "openai_compatible",
        max_retries: int = 3,
        fallback: PCUEngine | None = None,
    ):
        self.model = model
        self.provider = provider
        self.max_retries = max_retries
        self.fallback = fallback or PCUEngine()
        self.response_records: list[dict[str, Any]] = []
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("JUDGE_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY or JUDGE_API_KEY is required for --pcu-mode llm/hybrid")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("API_BASE") or os.getenv("JUDGE_BASE_URL") or None
        timeout = float(os.getenv("OPENAI_TIMEOUT_SEC", "120"))
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def extract_pcus(self, context_sources: list[ContextSource], metadata: dict[str, Any] | None = None) -> list[PCU]:
        metadata = metadata or {}
        prompt = build_pcu_prompt(context_sources, metadata)
        last_error: Exception | None = None
        messages = [
            {"role": "system", "content": PCU_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=messages,
                    response_format={"type": "json_object"},
                )
                self.response_records.append(
                    {
                        "kind": "pcu_annotation",
                        "model": self.model,
                        "response_id": getattr(response, "id", None),
                        "prompt_version": PCU_PROMPT_VERSION,
                        "instance_id": metadata.get("instance_id"),
                        "attempt": attempt + 1,
                    }
                )
                content = response.choices[0].message.content or "{}"
                obj = parse_json_object(content)
                pcus = self._pcus_from_llm_json(obj, context_sources, metadata)
                if pcus:
                    return pcus
                raise ValueError("LLM returned no grounded PCUs")
            except Exception as exc:
                last_error = exc
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The previous PCU annotation failed validation: {exc}. "
                            "Return valid JSON with grounded evidence_quote values copied exactly from source text."
                        ),
                    }
                )
                time.sleep(1.0 * (attempt + 1))

        pcus = self.fallback.extract_pcus(context_sources, metadata)
        for pcu in pcus:
            pcu.description = f"[fallback_after_llm_error={type(last_error).__name__}] {pcu.description}"
        return pcus

    def _pcus_from_llm_json(
        self,
        obj: dict[str, Any],
        context_sources: list[ContextSource],
        metadata: dict[str, Any],
    ) -> list[PCU]:
        raw_pcus = obj.get("pcus", [])
        if not isinstance(raw_pcus, list):
            raise ValueError("Expected top-level field `pcus` to be a list")
        by_ref = {source.ref_id: source for source in context_sources}
        by_source = {}
        for source in context_sources:
            by_source.setdefault(source.source, []).append(source)

        pcus: list[PCU] = []
        used: set[str] = set()
        for idx, raw in enumerate(raw_pcus[:10], start=1):
            if not isinstance(raw, dict):
                continue
            quote = str(raw.get("evidence_quote") or raw.get("quote") or raw.get("text") or "").strip()
            source_hint = str(raw.get("source") or "").strip()
            ref_hint = str(raw.get("ref_id") or "").strip()
            span = ground_quote_to_span(quote, source_hint, ref_hint, context_sources, by_ref, by_source)
            if span is None:
                continue
            pcu_id = str(raw.get("pcu_id") or "").strip()
            if not pcu_id or pcu_id in used:
                pcu_id = stable_id("PCU", metadata.get("instance_id", ""), span.ref_id, span.start, span.end, quote[:80], length=8)
            used.add(pcu_id)
            necessity = str(raw.get("necessity") or "soft").strip().lower()
            if necessity not in {"hard", "soft"}:
                necessity = "soft"
            importance = float(raw.get("importance", 0.0) or 0.0)
            if importance <= 0:
                importance = 0.85 if necessity == "hard" else 0.45
            pcu = PCU(
                pcu_id=pcu_id,
                necessity=necessity,  # type: ignore[arg-type]
                source_spans=[span],
                expected_patch_effect=str(raw.get("expected_patch_effect") or raw.get("patch_effect") or "").strip(),
                description=str(raw.get("description") or raw.get("rationale") or "").strip(),
                importance=max(0.0, min(1.0, importance)),
                ablation=AblationEvidence(
                    full_success_rate=0.0,
                    masked_success_rate=0.0,
                    success_delta=0.0,
                    trials=0,
                    method="llm_annotation_pending_ablation",
                ),
            )
            if not pcu.expected_patch_effect:
                pcu.expected_patch_effect = f"Patch should reflect this causal constraint: {quote[:240]}"
            pcus.append(pcu)

        pcus.sort(key=lambda p: (p.necessity != "hard", -p.importance, p.pcu_id))
        return pcus


class HybridPCUEngine:
    """Use LLM PCUs when possible and merge in heuristic oracle-patch PCUs."""

    def __init__(self, llm_engine: LLMPCUEngine, heuristic_engine: PCUEngine | None = None):
        self.llm_engine = llm_engine
        self.heuristic_engine = heuristic_engine or PCUEngine()

    @property
    def response_records(self) -> list[dict[str, Any]]:
        return self.llm_engine.response_records

    def extract_pcus(self, context_sources: list[ContextSource], metadata: dict[str, Any] | None = None) -> list[PCU]:
        metadata = metadata or {}
        llm_pcus = self.llm_engine.extract_pcus(context_sources, metadata)
        heuristic = self.heuristic_engine.extract_pcus(context_sources, metadata)
        merged = list(llm_pcus)
        seen = {(p.source_spans[0].ref_id, p.source_spans[0].start, p.source_spans[0].end) for p in merged if p.source_spans}
        for pcu in heuristic:
            if pcu.source_spans and pcu.source_spans[0].source == "patch":
                key = (pcu.source_spans[0].ref_id, pcu.source_spans[0].start, pcu.source_spans[0].end)
                if key not in seen:
                    pcu.description = "[heuristic_patch_oracle] " + pcu.description
                    merged.append(pcu)
        merged.sort(key=lambda p: (p.necessity != "hard", -p.importance, p.pcu_id))
        return merged[:10]


def build_pcu_prompt(context_sources: list[ContextSource], metadata: dict[str, Any]) -> str:
    packed = []
    for source in context_sources:
        if source.source == "patch":
            max_chars = 7000
        elif source.source in {"logs", "tests"}:
            max_chars = 5000
        else:
            max_chars = 3500
        packed.append(
            {
                "ref_id": source.ref_id,
                "source": source.source,
                "visible_to_agent": source.visible_to_agent,
                "metadata": source.metadata,
                "text": truncate_text(source.text, max_chars),
            }
        )
    payload = {
        "instance_id": metadata.get("instance_id"),
        "repo": metadata.get("repo"),
        "base_commit": metadata.get("base_commit"),
        "instruction": (
            "Identify 3-8 PCUs for this case. Each PCU must be case-specific. "
            "Prefer issue/test/log facts for hard PCUs; use gold patch semantics as soft PCUs unless they encode an indispensable behavior. "
            "For every PCU, copy evidence_quote exactly from one context source so the script can ground it to character offsets."
        ),
        "output_schema": {
            "pcus": [
                {
                    "pcu_id": "optional stable id",
                    "necessity": "hard|soft",
                    "source": "issue|logs|tests|patch|discussion|snippets",
                    "ref_id": "context source ref_id if known",
                    "evidence_quote": "exact substring copied from the source text",
                    "description": "why this is causally important",
                    "expected_patch_effect": "semantic behavior the final patch must implement",
                    "importance": 0.0,
                }
            ]
        },
        "context_sources": packed,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object")
    return obj


def ground_quote_to_span(
    quote: str,
    source_hint: str,
    ref_hint: str,
    context_sources: list[ContextSource],
    by_ref: dict[str, ContextSource],
    by_source: dict[str, list[ContextSource]],
) -> SourceSpan | None:
    candidates: list[ContextSource] = []
    if ref_hint and ref_hint in by_ref:
        candidates.append(by_ref[ref_hint])
    if source_hint:
        candidates.extend(s for s in by_source.get(source_hint, []) if s not in candidates)
    candidates.extend(s for s in context_sources if s not in candidates)

    quote = quote.strip()
    if not quote:
        return None
    for source in candidates:
        start = source.text.find(quote)
        if start >= 0:
            return SourceSpan(source=source.source, ref_id=source.ref_id, start=start, end=start + len(quote))

    normalized_quote = normalize_for_match(quote)
    best: tuple[float, ContextSource, int, int] | None = None
    for source in candidates:
        source_text = source.text
        window_size = min(len(source_text), max(len(quote) * 2, 160))
        step = max(40, window_size // 4)
        for start in range(0, max(1, len(source_text) - window_size + 1), step):
            end = min(len(source_text), start + window_size)
            window = source_text[start:end]
            score = SequenceMatcher(None, normalized_quote, normalize_for_match(window)).ratio()
            if best is None or score > best[0]:
                best = (score, source, start, end)
    if best and best[0] >= 0.72:
        _, source, start, end = best
        return SourceSpan(source=source.source, ref_id=source.ref_id, start=start, end=end)
    return None


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
