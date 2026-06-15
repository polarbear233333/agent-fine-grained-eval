from __future__ import annotations

from typing import Any

from .schema import BenchmarkCase, ContextPlan, ContextPlanItem, ContextState, MemoryItem, VisibleContextItem
from .utils import approx_tokens, truncate_text


class ContextManager:
    """Apply keep/summarize/discard/memory plans under a token budget."""

    def __init__(self, case: BenchmarkCase):
        self.case = case
        self.state = ContextState()
        self.memory_history: list[list[MemoryItem]] = []

    def apply_plan(self, raw_plan: dict[str, Any] | ContextPlan, turn_id: int | None = None) -> ContextState:
        plan = self._normalize_plan(raw_plan, turn_id=turn_id)
        previous_pcus = {m.pcu_id for m in self.state.memory if m.pcu_id}
        memory = self._merge_memory(plan.memory)
        current_pcus = {m.pcu_id for m in memory if m.pcu_id}

        visible_items: list[VisibleContextItem] = []
        discarded_refs = set(plan.discard)
        summary_by_ref = {item.ref: item.summary for item in plan.summarize}

        for mem in memory:
            visible_items.append(
                VisibleContextItem(
                    ref_id=f"memory:{mem.key}",
                    source="memory",
                    text=mem.value,
                    token_count=approx_tokens(mem.value),
                    operation="memory",
                )
            )

        for source in self.case.context_sources:
            if not source.visible_to_agent:
                discarded_refs.add(source.ref_id)
                continue
            if source.ref_id in discarded_refs:
                continue
            if source.ref_id in plan.keep:
                visible_items.append(
                    VisibleContextItem(
                        ref_id=source.ref_id,
                        source=source.source,
                        text=source.text,
                        token_count=source.token_count,
                        operation="keep",
                    )
                )
            elif source.ref_id in summary_by_ref:
                summary = summary_by_ref[source.ref_id] or source.text[:240]
                visible_items.append(
                    VisibleContextItem(
                        ref_id=source.ref_id,
                        source=source.source,
                        text=summary,
                        token_count=approx_tokens(summary),
                        operation="summarize",
                    )
                )
            elif source.source == "issue":
                visible_items.append(
                    VisibleContextItem(
                        ref_id=source.ref_id,
                        source=source.source,
                        text=source.text,
                        token_count=source.token_count,
                        operation="implicit",
                    )
                )
            else:
                discarded_refs.add(source.ref_id)

        visible_items, overflow = self._fit_budget(visible_items)
        state = ContextState(
            visible_items=visible_items,
            memory=memory,
            visible_tokens=sum(item.token_count for item in visible_items),
            discarded_refs=sorted(discarded_refs),
            forgotten_pcus=sorted(p for p in previous_pcus - current_pcus if p),
            budget_overflow_tokens=overflow,
        )
        self.state = state
        self.memory_history.append(memory)
        return state

    def _normalize_plan(self, raw_plan: dict[str, Any] | ContextPlan, turn_id: int | None) -> ContextPlan:
        if isinstance(raw_plan, ContextPlan):
            plan = raw_plan
        else:
            payload = raw_plan.get("context_plan", raw_plan) if isinstance(raw_plan, dict) else {}
            summarize = []
            for item in payload.get("summarize", []) or []:
                if isinstance(item, str):
                    summarize.append({"ref": item, "summary": ""})
                elif isinstance(item, dict):
                    summarize.append({"ref": str(item.get("ref", "")), "summary": str(item.get("summary", ""))})
            memory = []
            for item in payload.get("memory", []) or []:
                if isinstance(item, str):
                    memory.append({"key": item[:60], "value": item})
                elif isinstance(item, dict):
                    memory.append(
                        {
                            "key": str(item.get("key") or item.get("pcu_id") or item.get("name") or "memory"),
                            "value": str(item.get("value") or item.get("summary") or item.get("text") or ""),
                            "importance": float(item.get("importance", 0.5) or 0.5),
                            "pcu_id": item.get("pcu_id"),
                            "turn_id": item.get("turn_id", turn_id),
                        }
                    )
            plan = ContextPlan(
                keep=[str(x) for x in payload.get("keep", []) or [] if x],
                summarize=[ContextPlanItem(**x) for x in summarize if x.get("ref")],
                discard=[str(x) for x in payload.get("discard", []) or [] if x],
                memory=[MemoryItem(**x) for x in memory if x.get("value")],
            )
        if turn_id is not None:
            for item in plan.memory:
                if item.turn_id is None:
                    item.turn_id = turn_id
        return plan

    def _merge_memory(self, new_items: list[MemoryItem]) -> list[MemoryItem]:
        by_key: dict[str, MemoryItem] = {item.key: item for item in self.state.memory}
        for item in new_items:
            existing = by_key.get(item.key)
            if existing is None or item.importance >= existing.importance or item.value != existing.value:
                by_key[item.key] = item
        ranked = sorted(
            by_key.values(),
            key=lambda m: (-(m.importance or 0.0), -(m.turn_id or -1), m.key),
        )
        return ranked[: max(0, self.case.context_budget.memory_slots)]

    def _fit_budget(self, items: list[VisibleContextItem]) -> tuple[list[VisibleContextItem], int]:
        budget = max(1, self.case.context_budget.max_visible_tokens)
        total = sum(item.token_count for item in items)
        if total <= budget:
            return items, 0

        overflow = total - budget
        priority = {"memory": 0, "keep": 1, "summarize": 2, "implicit": 3}
        ranked = sorted(items, key=lambda item: (priority.get(item.operation, 9), item.token_count))
        kept: list[VisibleContextItem] = []
        used = 0
        for item in ranked:
            if used + item.token_count <= budget:
                kept.append(item)
                used += item.token_count
                continue
            remaining = budget - used
            if item.operation in {"memory", "summarize", "implicit"} and remaining > 16:
                max_chars = remaining * 4
                shortened = truncate_text(item.text, max_chars)
                kept.append(
                    VisibleContextItem(
                        ref_id=item.ref_id,
                        source=item.source,
                        text=shortened,
                        token_count=approx_tokens(shortened),
                        operation=item.operation,
                    )
                )
                used += approx_tokens(shortened)
            break
        original_order = {item.ref_id: i for i, item in enumerate(items)}
        kept.sort(key=lambda item: original_order.get(item.ref_id, 999999))
        return kept, overflow

    def visible_text(self) -> str:
        chunks = []
        for item in self.state.visible_items:
            chunks.append(f"[{item.source}:{item.ref_id}]\n{item.text}")
        return "\n\n".join(chunks)

