from agent_eval.context_manager import ContextManager
from agent_eval.dataset_builder import DatasetBuilder
from agent_eval.evaluation_engine import EvaluationEngine
from agent_eval.schema import BenchmarkCase, ContextBudget, ContextSource, PCU, SourceSpan


def test_pcu_required_shape_and_issue_extraction():
    row = {
        "instance_id": "demo__repo-1",
        "repo": "demo/repo",
        "problem_statement": "Widget.render should preserve raw HTML when autoescape is disabled. Actual output is escaped.",
        "patch": "diff --git a/widget.py b/widget.py\n+++ b/widget.py\n@@\n+return render(value, autoescape=self.autoescape)\n",
    }
    case = DatasetBuilder(track="A").build_case(row)
    hard = [p for p in case.pcus if p.necessity == "hard"]
    assert hard
    public = hard[0].required_shape()
    assert set(public) == {"pcu_id", "necessity", "source_spans", "expected_patch_effect"}
    assert public["source_spans"][0]["source"] == "issue"


def test_context_manager_applies_budget_and_memory_slots():
    case = BenchmarkCase(
        instance_id="x",
        context_sources=[
            ContextSource(ref_id="issue-1", source="issue", text="A should do B.", token_count=4),
            ContextSource(ref_id="noise-1", source="discussion", text="noise " * 200, token_count=200),
        ],
        context_budget=ContextBudget(max_visible_tokens=30, memory_slots=1),
        pcus=[
            PCU(
                pcu_id="PCU-1",
                necessity="hard",
                source_spans=[SourceSpan(source="issue", ref_id="issue-1", start=0, end=12)],
                expected_patch_effect="Do B.",
            )
        ],
    )
    manager = ContextManager(case)
    state = manager.apply_plan(
        {
            "keep": ["issue-1", "noise-1"],
            "memory": [
                {"key": "low", "value": "low", "importance": 0.1, "pcu_id": "PCU-low"},
                {"key": "high", "value": "remember Do B", "importance": 0.9, "pcu_id": "PCU-1"},
            ],
        }
    )
    assert len(state.memory) == 1
    assert state.memory[0].pcu_id == "PCU-1"
    assert state.visible_tokens <= 30


def test_evaluation_scores_pcu_recall():
    case = BenchmarkCase(
        instance_id="x",
        context_sources=[ContextSource(ref_id="issue-1", source="issue", text="A should do B.", token_count=4)],
        pcus=[
            PCU(
                pcu_id="PCU-1",
                necessity="hard",
                source_spans=[SourceSpan(source="issue", ref_id="issue-1", start=0, end=12)],
                expected_patch_effect="Do B.",
            )
        ],
    )
    manager = ContextManager(case)
    state = manager.apply_plan({"keep": ["issue-1"], "memory": [{"key": "PCU-1", "pcu_id": "PCU-1", "value": "Do B."}]})
    metrics = EvaluationEngine().score_case(case, {"context_plan": {"keep": ["issue-1"]}}, state)
    assert metrics.hard_pcu_recall == 1.0

