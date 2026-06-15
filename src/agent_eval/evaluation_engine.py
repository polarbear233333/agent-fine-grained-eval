from __future__ import annotations

from statistics import mean
from typing import Any

from .pcu_engine import pcu_keywords, pcu_minimal_tokens
from .schema import BenchmarkCase, CaseMetrics, ContextState, PCU
from .utils import approx_tokens, keyword_set, write_json


class EvaluationEngine:
    def score_case(
        self,
        case: BenchmarkCase,
        agent_output: dict[str, Any],
        state: ContextState,
        task_success: bool | None = None,
        final_patch: str = "",
        actions: list[dict[str, Any]] | None = None,
    ) -> CaseMetrics:
        actions = actions or agent_output.get("actions", []) or []
        refs_text = self._refs_text(agent_output, state)
        retained_text = refs_text + "\n" + "\n".join(item.text for item in state.visible_items)
        hard = [p for p in case.pcus if p.necessity == "hard"]
        soft = [p for p in case.pcus if p.necessity == "soft"]
        hard_recall = self._recall(hard, retained_text, state)
        soft_recall = self._recall(soft, retained_text, state) if soft else 1.0
        pcu_at_k = self._recall(sorted(case.pcus, key=lambda p: -p.importance)[: max(1, case.context_budget.memory_slots)], retained_text, state)
        minimal_tokens = pcu_minimal_tokens(case.pcus)
        cbr = state.visible_tokens / max(1, minimal_tokens)
        alignment = self._patch_alignment(case.pcus, final_patch or str(agent_output.get("final_patch") or ""))
        retrieval_cost = self._retrieval_cost(case, actions)
        delayed = self._delayed_recall(case, retained_text, state)
        conflict = self._conflict_resolution(case, retained_text)

        notes = []
        if state.budget_overflow_tokens > 0:
            notes.append(f"budget_overflow_tokens={state.budget_overflow_tokens}")
        if not final_patch and case.track in {"budgeted_patch", "long_horizon"}:
            notes.append("no_final_patch_available")

        return CaseMetrics(
            instance_id=case.instance_id,
            track=case.track,
            task_success=task_success,
            pcu_recall_at_k=round(pcu_at_k, 4),
            hard_pcu_recall=round(hard_recall, 4),
            soft_pcu_recall=round(soft_recall, 4),
            pcu_to_patch_alignment=round(alignment, 4),
            context_bloat_ratio=round(cbr, 4),
            retrieval_cost=round(retrieval_cost, 4),
            delayed_pcu_recall=round(delayed, 4),
            forgetting_events=len(state.forgotten_pcus),
            conflict_resolution_accuracy=round(conflict, 4),
            retained_tokens=state.visible_tokens,
            minimal_hard_pcu_tokens=minimal_tokens,
            notes=notes,
        )

    def aggregate(self, metrics: list[CaseMetrics]) -> dict[str, Any]:
        if not metrics:
            return {"case_count": 0}
        task_values = [1.0 if m.task_success else 0.0 for m in metrics if m.task_success is not None]
        return {
            "case_count": len(metrics),
            "task_success": round(mean(task_values), 4) if task_values else None,
            "pcu_recall_at_k": round(mean(m.pcu_recall_at_k for m in metrics), 4),
            "hard_pcu_recall": round(mean(m.hard_pcu_recall for m in metrics), 4),
            "soft_pcu_recall": round(mean(m.soft_pcu_recall for m in metrics), 4),
            "pcu_to_patch_alignment": round(mean(m.pcu_to_patch_alignment for m in metrics), 4),
            "context_bloat_ratio": round(mean(m.context_bloat_ratio for m in metrics), 4),
            "retrieval_cost": round(mean(m.retrieval_cost for m in metrics), 4),
            "delayed_pcu_recall": round(mean(m.delayed_pcu_recall for m in metrics), 4),
            "forgetting_events": round(mean(m.forgetting_events for m in metrics), 4),
            "conflict_resolution_accuracy": round(mean(m.conflict_resolution_accuracy for m in metrics), 4),
        }

    def write_outputs(self, run_dir: str, metrics: list[CaseMetrics], config: dict[str, Any]) -> None:
        aggregate = self.aggregate(metrics)
        write_json(
            f"{run_dir}/metrics.json",
            {
                "config": config,
                "aggregate": aggregate,
                "cases": [m.model_dump() for m in metrics],
            },
        )
        lines = [
            "# Experiment Summary",
            "",
            f"- cases: {aggregate.get('case_count', 0)}",
            f"- task_success: {aggregate.get('task_success')}",
            f"- hard_pcu_recall: {aggregate.get('hard_pcu_recall')}",
            f"- pcu_recall_at_k: {aggregate.get('pcu_recall_at_k')}",
            f"- pcu_to_patch_alignment: {aggregate.get('pcu_to_patch_alignment')}",
            f"- context_bloat_ratio: {aggregate.get('context_bloat_ratio')}",
            f"- delayed_pcu_recall: {aggregate.get('delayed_pcu_recall')}",
            f"- conflict_resolution_accuracy: {aggregate.get('conflict_resolution_accuracy')}",
            "",
            "This run is reproducible from the command and config stored in metrics.json.",
        ]
        with open(f"{run_dir}/summary.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _refs_text(self, agent_output: dict[str, Any], state: ContextState) -> str:
        plan = agent_output.get("context_plan", agent_output)
        chunks: list[str] = []
        for key in ["keep", "discard"]:
            chunks.extend(str(x) for x in plan.get(key, []) or [])
        for item in plan.get("summarize", []) or []:
            chunks.append(str(item))
        for item in plan.get("memory", []) or []:
            chunks.append(str(item))
        for item in state.visible_items:
            chunks.append(item.ref_id)
        for mem in state.memory:
            chunks.append(f"{mem.key} {mem.value} {mem.pcu_id or ''}")
        return "\n".join(chunks).lower()

    def _recall(self, pcus: list[PCU], retained_text: str, state: ContextState) -> float:
        if not pcus:
            return 1.0
        covered = 0
        retained = retained_text.lower()
        refs = {item.ref_id for item in state.visible_items}
        memory_pcus = {m.pcu_id for m in state.memory if m.pcu_id}
        for pcu in pcus:
            if pcu.pcu_id in memory_pcus or pcu.pcu_id.lower() in retained:
                covered += 1
                continue
            span_hit = False
            for span in pcu.source_spans:
                if span.ref_id and span.ref_id in refs:
                    span_hit = True
                    break
                if span.ref_id and span.ref_id.lower() in retained:
                    span_hit = True
                    break
            if span_hit:
                covered += 1
                continue
            keys = pcu_keywords(pcu)
            if keys and len(keys & keyword_set(retained, max_words=200)) >= min(3, len(keys)):
                covered += 1
        return covered / len(pcus)

    def _patch_alignment(self, pcus: list[PCU], final_patch: str) -> float:
        if not final_patch.strip():
            return 0.0
        patch_keys = keyword_set(final_patch, max_words=300)
        hard = [p for p in pcus if p.necessity == "hard"] or pcus
        if not hard:
            return 0.0
        aligned = 0
        for pcu in hard:
            keys = keyword_set(pcu.expected_patch_effect + " " + pcu.description, max_words=40)
            if not keys:
                continue
            if len(keys & patch_keys) >= min(2, len(keys)):
                aligned += 1
        return aligned / max(1, len(hard))

    def _retrieval_cost(self, case: BenchmarkCase, actions: list[dict[str, Any]]) -> float:
        cost_model = case.context_budget.cost_model
        total = 0.0
        for action in actions:
            if not isinstance(action, dict):
                total += cost_model.get("external_retrieve", 1.5)
                continue
            action_type = str(action.get("type") or action.get("action") or "external_retrieve")
            base = cost_model.get(action_type, cost_model.get("external_retrieve", 1.5))
            text_mass = approx_tokens(str(action.get("result") or action.get("content") or action.get("query") or "")) / 1000.0
            total += base + text_mass
        return total

    def _delayed_recall(self, case: BenchmarkCase, retained_text: str, state: ContextState) -> float:
        delayed_pcus = set()
        for turn in case.interaction_script:
            if "delayed_dependency" in turn.tags or turn.turn_id >= 6:
                delayed_pcus.update(turn.introduced_pcus)
        if not delayed_pcus:
            hard = [p.pcu_id for p in case.pcus if p.necessity == "hard"]
            delayed_pcus.update(hard[:1])
        selected = [p for p in case.pcus if p.pcu_id in delayed_pcus]
        return self._recall(selected, retained_text, state) if selected else 1.0

    def _conflict_resolution(self, case: BenchmarkCase, retained_text: str) -> float:
        has_conflict = any("conflict" in tag for turn in case.interaction_script for tag in turn.tags)
        if not has_conflict:
            return 1.0
        hard_text = " ".join(p.pcu_id + " " + p.expected_patch_effect for p in case.pcus if p.necessity == "hard").lower()
        retained = retained_text.lower()
        hard_hit = any(token in retained for token in keyword_set(hard_text, max_words=30))
        low_confidence_noise = "low confidence" in retained or "note a" in retained
        if hard_hit and not low_confidence_noise:
            return 1.0
        if hard_hit:
            return 0.75
        return 0.0

