from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .ce_schema import ContextEngineeringInstance, ContextSource, ContextBudget, InteractionTurn, PCU, SourceSpan, ExpectedPatchEffect
from .swebench_loader import normalize_case
from .utils import read_jsonl, write_jsonl, ensure_dir


def approx_tokens(text: str) -> int:
    # Conservative multilingual approximation. Good enough for budgeting experiments.
    return max(1, len(text) // 4)


def make_span_id(prefix: str, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{prefix}-{h}"


def split_patch_files(patch: str) -> List[str]:
    files = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                files.append(parts[3].removeprefix("b/"))
    return files[:5]


def build_pcus(case: Dict[str, Any], context_sources: List[ContextSource]) -> List[PCU]:
    issue_ref = next((s.ref_id for s in context_sources if s.source == "issue"), "issue-0")
    test_ref = next((s.ref_id for s in context_sources if s.source == "tests"), None)
    patch_files = split_patch_files(case.get("patch") or "")
    expected_file = patch_files[0] if patch_files else None
    issue_text = case.get("problem_statement", "")
    short_issue = issue_text[:240].replace("\n", " ")
    pcus = [
        PCU(
            pcu_id="PCU-issue-core",
            necessity="hard",
            description=f"Core user-reported bug/request: {short_issue}",
            source_spans=[SourceSpan(source="issue", ref_id=issue_ref, start=0, end=min(len(issue_text), 600))],
            expected_patch_effect=ExpectedPatchEffect(file=expected_file, semantic_change="The final patch should directly address the issue's required behavior."),
        )
    ]
    if test_ref:
        pcus.append(
            PCU(
                pcu_id="PCU-test-signal",
                necessity="soft",
                description="Test patch or failing-test signal that constrains the expected behavior.",
                source_spans=[SourceSpan(source="tests", ref_id=test_ref, start=0, end=600)],
                expected_patch_effect=ExpectedPatchEffect(file=expected_file, semantic_change="The implementation should satisfy the provided or implied tests."),
            )
        )
    return pcus


def inject_noise_turns(case: Dict[str, Any], pcus: List[PCU], noise_turns: int, seed: int) -> List[InteractionTurn]:
    rnd = random.Random(seed)
    issue = case.get("problem_statement", "")
    repo = case.get("repo") or "unknown/repo"
    plausible_files = split_patch_files(case.get("patch") or "") or ["utils.py", "core.py", "tests/test_regression.py"]
    noise_templates = [
        "Maybe this is mostly a style/refactor issue; consider renaming variables before changing behavior.",
        "A teammate mentioned `{file}` might be related, but this is not verified.",
        "Long log excerpt: INFO build started ... DEBUG unrelated cache hit ... WARNING optional dependency missing ... INFO continue ...",
        "Potentially conflicting suggestion: maybe returning None is acceptable here, but check the issue/tests before trusting this.",
        "Repeated reminder: avoid overloading the context with entire files unless a span is clearly relevant.",
    ]
    turns = [InteractionTurn(turn_id=1, role="user", content=f"We need to fix this SWE-bench issue in {repo}:\n\n{issue}", tags=["issue"], introduced_pcus=[pcus[0].pcu_id] if pcus else [])]
    for i in range(noise_turns):
        tmpl = rnd.choice(noise_templates)
        turns.append(InteractionTurn(turn_id=i + 2, role="user", content=tmpl.format(file=rnd.choice(plausible_files)), tags=["noise" if i % 3 else "possible_hint"]))
    if len(pcus) > 1:
        turns.append(InteractionTurn(turn_id=len(turns)+1, role="user", content="Later checkpoint: use the test/failure signal rather than only the earlier vague discussion.", tags=["delayed_dependency"], introduced_pcus=[pcus[1].pcu_id]))
    return turns


def convert_case_to_ce(row: Dict[str, Any], track: str = "context_management", max_visible_tokens: int = 8192, memory_slots: int = 8, noise_turns: int = 8, seed: int = 42) -> ContextEngineeringInstance:
    case = normalize_case(row)
    sources: List[ContextSource] = []
    for source_name, text in [
        ("issue", case.get("problem_statement") or ""),
        ("discussion", case.get("hints_text") or ""),
        ("patch", case.get("patch") or ""),
        ("tests", case.get("test_patch") or ""),
    ]:
        if text.strip():
            sources.append(ContextSource(ref_id=make_span_id(source_name, text), source=source_name, text=text, token_count=approx_tokens(text)))
    pcus = build_pcus(case, sources)
    turns = inject_noise_turns(case, pcus, noise_turns=noise_turns, seed=seed)
    return ContextEngineeringInstance(
        instance_id=case["instance_id"],
        track=track,  # type: ignore[arg-type]
        metadata={"repo": case.get("repo"), "base_commit": case.get("base_commit"), "source": "swebench-converted"},
        task={"issue": case.get("problem_statement"), "gold_patch_available": bool(case.get("patch")), "test_patch_available": bool(case.get("test_patch"))},
        context_sources=sources,
        interaction_script=turns,
        pcus=pcus,
        context_budget=ContextBudget(max_visible_tokens=max_visible_tokens, memory_slots=memory_slots),
        evaluation={"requires_context_plan": True, "requires_patch": track in {"budgeted_patch", "long_horizon"}},
    )


def convert_jsonl(input_path: str | Path, output_path: str | Path, **kwargs: Any) -> None:
    rows = []
    for row in read_jsonl(input_path):
        rows.append(convert_case_to_ce(row, **kwargs).model_dump())
    ensure_dir(Path(output_path).parent)
    write_jsonl(output_path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SWE-bench-style JSONL into Context Engineering benchmark JSONL.")
    parser.add_argument("--input", required=True, help="Input SWE-bench-style JSONL path")
    parser.add_argument("--output", required=True, help="Output CE JSONL path")
    parser.add_argument("--track", default="context_management", choices=["context_management", "budgeted_patch", "long_horizon"])
    parser.add_argument("--max-visible-tokens", type=int, default=8192)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--noise-turns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    convert_jsonl(args.input, args.output, track=args.track, max_visible_tokens=args.max_visible_tokens, memory_slots=args.memory_slots, noise_turns=args.noise_turns, seed=args.seed)
    print(f"Wrote Context Engineering dataset to {args.output}")


if __name__ == "__main__":
    main()
