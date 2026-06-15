from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


TrackName = Literal["context_management", "budgeted_patch", "long_horizon"]


class ContextSource(BaseModel):
    ref_id: str
    source: str
    text: str
    token_count: int = 0
    visible_to_agent: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceSpan(BaseModel):
    source: str
    start: int = 0
    end: int = 0
    ref_id: Optional[str] = None


class AblationEvidence(BaseModel):
    full_success_rate: float = 0.0
    masked_success_rate: float = 0.0
    success_delta: float = 0.0
    trials: int = 0
    method: str = "proxy"


class PCU(BaseModel):
    pcu_id: str
    necessity: Literal["hard", "soft"] = "soft"
    source_spans: list[SourceSpan] = Field(default_factory=list)
    expected_patch_effect: str = ""
    description: str = ""
    importance: float = 0.0
    ablation: Optional[AblationEvidence] = None

    def required_shape(self) -> dict[str, Any]:
        """Return the minimal public PCU schema requested by the benchmark."""
        return {
            "pcu_id": self.pcu_id,
            "necessity": self.necessity,
            "source_spans": [
                {"source": s.source, "start": s.start, "end": s.end}
                for s in self.source_spans
            ],
            "expected_patch_effect": self.expected_patch_effect,
        }


class ContextBudget(BaseModel):
    max_visible_tokens: int = 8192
    memory_slots: int = 8
    allowed_operations: list[str] = Field(
        default_factory=lambda: ["keep", "summarize", "discard", "external_retrieve"]
    )
    cost_model: dict[str, float] = Field(
        default_factory=lambda: {
            "keep": 1.0,
            "summarize": 0.3,
            "discard": 0.0,
            "external_retrieve": 1.5,
            "open_file": 1.0,
            "search": 0.8,
        }
    )


class InteractionTurn(BaseModel):
    turn_id: int
    role: Literal["system", "user", "agent", "evaluator"] = "user"
    content: str
    tags: list[str] = Field(default_factory=list)
    introduced_pcus: list[str] = Field(default_factory=list)
    visible_ref_ids: list[str] = Field(default_factory=list)


class BenchmarkCase(BaseModel):
    benchmark_id: str = "pcu-context-bench-v1"
    instance_id: str
    track: TrackName = "context_management"
    metadata: dict[str, Any] = Field(default_factory=dict)
    task: dict[str, Any] = Field(default_factory=dict)
    context_sources: list[ContextSource] = Field(default_factory=list)
    interaction_script: list[InteractionTurn] = Field(default_factory=list)
    pcus: list[PCU] = Field(default_factory=list)
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    evaluation: dict[str, Any] = Field(default_factory=dict)


class ContextPlanItem(BaseModel):
    ref: str
    summary: str = ""


class MemoryItem(BaseModel):
    key: str
    value: str
    importance: float = 0.5
    pcu_id: Optional[str] = None
    turn_id: Optional[int] = None


class ContextPlan(BaseModel):
    keep: list[str] = Field(default_factory=list)
    summarize: list[ContextPlanItem] = Field(default_factory=list)
    discard: list[str] = Field(default_factory=list)
    memory: list[MemoryItem] = Field(default_factory=list)


class VisibleContextItem(BaseModel):
    ref_id: str
    source: str
    text: str
    token_count: int
    operation: Literal["keep", "summarize", "memory", "implicit"] = "implicit"


class ContextState(BaseModel):
    visible_items: list[VisibleContextItem] = Field(default_factory=list)
    memory: list[MemoryItem] = Field(default_factory=list)
    visible_tokens: int = 0
    discarded_refs: list[str] = Field(default_factory=list)
    forgotten_pcus: list[str] = Field(default_factory=list)
    budget_overflow_tokens: int = 0


class RunnerResult(BaseModel):
    status: Literal["pass", "fail", "skipped", "error"] = "skipped"
    logs: str = ""
    diff: str = ""
    worktree: Optional[str] = None
    cleanup_performed: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseMetrics(BaseModel):
    instance_id: str
    track: TrackName
    task_success: Optional[bool] = None
    pcu_recall_at_k: float = 0.0
    hard_pcu_recall: float = 0.0
    soft_pcu_recall: float = 0.0
    pcu_to_patch_alignment: float = 0.0
    context_bloat_ratio: float = 0.0
    retrieval_cost: float = 0.0
    delayed_pcu_recall: float = 0.0
    forgetting_events: int = 0
    conflict_resolution_accuracy: float = 0.0
    retained_tokens: int = 0
    minimal_hard_pcu_tokens: int = 0
    notes: list[str] = Field(default_factory=list)

