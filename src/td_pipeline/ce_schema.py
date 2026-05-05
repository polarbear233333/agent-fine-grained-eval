from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ContextSource(BaseModel):
    ref_id: str
    source: Literal["issue", "discussion", "logs", "snippets", "patch", "tests", "metadata", "other"] = "other"
    text: str
    token_count: int = 0


class SourceSpan(BaseModel):
    source: str
    ref_id: str
    start: int = 0
    end: int = 0


class ExpectedPatchEffect(BaseModel):
    file: Optional[str] = None
    semantic_change: str = ""


class PCU(BaseModel):
    pcu_id: str
    necessity: Literal["hard", "soft"] = "hard"
    description: str
    source_spans: List[SourceSpan] = Field(default_factory=list)
    expected_patch_effect: ExpectedPatchEffect = Field(default_factory=ExpectedPatchEffect)


class ContextBudget(BaseModel):
    max_visible_tokens: int = 8192
    memory_slots: int = 8
    allowed_operations: List[str] = Field(default_factory=lambda: ["keep", "summarize", "discard", "external_retrieve"])
    cost_model: Dict[str, float] = Field(default_factory=lambda: {"keep": 1.0, "summarize": 0.3, "discard": 0.0, "external_retrieve": 1.5})


class InteractionTurn(BaseModel):
    turn_id: int
    role: Literal["user", "agent", "system", "evaluator"] = "user"
    content: str
    tags: List[str] = Field(default_factory=list)
    introduced_pcus: List[str] = Field(default_factory=list)


class ContextEngineeringInstance(BaseModel):
    benchmark_id: str = "swece-v1"
    instance_id: str
    track: Literal["context_management", "budgeted_patch", "long_horizon"] = "context_management"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    task: Dict[str, Any] = Field(default_factory=dict)
    context_sources: List[ContextSource] = Field(default_factory=list)
    interaction_script: List[InteractionTurn] = Field(default_factory=list)
    pcus: List[PCU] = Field(default_factory=list)
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    evaluation: Dict[str, Any] = Field(default_factory=dict)


class ContextPlanItem(BaseModel):
    ref: str
    summary: Optional[str] = None


class ContextPlan(BaseModel):
    keep: List[str] = Field(default_factory=list)
    summarize: List[ContextPlanItem] = Field(default_factory=list)
    discard: List[str] = Field(default_factory=list)
    memory: List[Dict[str, str]] = Field(default_factory=list)


class ContextEngineeringScore(BaseModel):
    hard_pcu_recall: float = 0.0
    soft_pcu_recall: float = 0.0
    context_bloat_ratio: float = 0.0
    memory_utilization: float = 0.0
    delayed_recall_accuracy: float = 0.0
    conflict_resolution_accuracy: float = 0.0
    noise_resistance_score: float = 0.0
    forgetting_events_count: int = 0
    actionable_recall_score: float = 0.0
    final_ce_score: float = 0.0
