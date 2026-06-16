from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Iterator

from .llm_pcu_engine import HybridPCUEngine, LLMPCUEngine
from .pcu_engine import PCUEngine
from .schema import BenchmarkCase, ContextBudget, ContextSource, InteractionTurn, TrackName
from .utils import (
    approx_tokens,
    append_jsonl,
    ensure_dir,
    extract_paths,
    extract_pr_description,
    first_nonempty,
    patch_files,
    read_jsonl_stream,
    stable_id,
    truncate_text,
)


TRACK_ALIASES = {
    "A": "context_management",
    "B": "budgeted_patch",
    "C": "long_horizon",
    "context_management": "context_management",
    "budgeted_patch": "budgeted_patch",
    "long_horizon": "long_horizon",
}


class DatasetBuilder:
    def __init__(
        self,
        track: str = "A",
        max_visible_tokens: int = 8192,
        memory_slots: int = 8,
        noise_turns: int | None = None,
        seed: int = 42,
        pcu_mode: str = "heuristic",
        pcu_model: str = "gpt-5.4-mini",
    ):
        self.track: TrackName = TRACK_ALIASES.get(track, track)  # type: ignore[assignment]
        self.max_visible_tokens = max_visible_tokens
        self.memory_slots = memory_slots
        self.noise_turns = noise_turns
        self.seed = seed
        heuristic_engine = PCUEngine()
        if pcu_mode == "heuristic":
            self.pcu_engine = heuristic_engine
        elif pcu_mode == "llm":
            self.pcu_engine = LLMPCUEngine(model=pcu_model, fallback=heuristic_engine)
        elif pcu_mode == "hybrid":
            self.pcu_engine = HybridPCUEngine(LLMPCUEngine(model=pcu_model, fallback=heuristic_engine), heuristic_engine)
        else:
            raise ValueError(f"Unsupported pcu_mode: {pcu_mode}")
        self.pcu_mode = pcu_mode
        self.pcu_model = pcu_model

    def iter_rows(
        self,
        dataset: str,
        input_paths: list[str] | None = None,
        hf_name: str | None = None,
        split: str = "test",
        max_cases: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        if dataset in {"local", "jsonl", "trajectories"}:
            if not input_paths:
                raise ValueError("--input is required for local/jsonl/trajectories datasets")
            yielded = 0
            for raw_path in input_paths:
                path = Path(raw_path)
                per_file_limit = None if max_cases is None else max_cases - yielded
                for row in read_jsonl_stream(path, limit=per_file_limit):
                    row["_source_file"] = str(path)
                    yield row
                    yielded += 1
                    if max_cases is not None and yielded >= max_cases:
                        return
            return

        if dataset in {"swebench", "swebench_lite", "swebench_verified", "swepro"}:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError("Install datasets to load SWE-bench from Hugging Face: pip install -e .[swebench]") from exc
            if dataset == "swebench_verified":
                name = hf_name or "SWE-bench/SWE-bench_Verified"
            elif dataset == "swepro":
                name = hf_name or "ScaleAI/SWE-bench_Pro"
            elif dataset == "swebench":
                name = hf_name or "SWE-bench/SWE-bench"
            else:
                name = hf_name or "SWE-bench/SWE-bench_Lite"
            ds = load_dataset(name, split=split)
            for idx, row in enumerate(ds):
                if max_cases is not None and idx >= max_cases:
                    break
                item = dict(row)
                item["_source_file"] = name
                yield item
            return

        raise ValueError(f"Unsupported dataset: {dataset}")

    def build_case(self, row: dict[str, Any], index: int = 0) -> BenchmarkCase:
        normalized = normalize_case(row)
        sources = build_context_sources(normalized)
        pcus = self.pcu_engine.extract_pcus(sources, metadata=normalized)
        script = build_interaction_script(
            normalized,
            sources,
            pcus,
            track=self.track,
            seed=self.seed + index,
            noise_turns=self.noise_turns,
        )
        budget = ContextBudget(
            max_visible_tokens=self.max_visible_tokens,
            memory_slots=self.memory_slots,
        )
        return BenchmarkCase(
            instance_id=normalized["instance_id"],
            track=self.track,
            metadata={
                "repo": normalized.get("repo"),
                "base_commit": normalized.get("base_commit"),
                "dataset_source": normalized.get("_source_file"),
                "source_kind": normalized.get("_source_kind"),
                "trajectory_success": normalized.get("success"),
                "observed_final_patch": normalized.get("observed_final_patch", ""),
                "detected_paths": normalized.get("detected_paths", []),
                "pcu_mode": self.pcu_mode,
                "pcu_model": self.pcu_model if self.pcu_mode in {"llm", "hybrid"} else None,
            },
            task={
                "problem_statement": normalized.get("problem_statement", ""),
                "hints_text": normalized.get("hints_text", ""),
                "requires_patch": self.track in {"budgeted_patch", "long_horizon"},
                "requires_context_plan": True,
            },
            context_sources=sources,
            interaction_script=script,
            pcus=pcus,
            context_budget=budget,
            evaluation={
                "track": self.track,
                "task_success_required": self.track in {"budgeted_patch", "long_horizon"},
                "pcu_required_shape": [pcu.required_shape() for pcu in pcus],
                "pcu_mode": self.pcu_mode,
            },
        )

    def build_file(
        self,
        dataset: str,
        output_path: str | Path,
        input_paths: list[str] | None = None,
        hf_name: str | None = None,
        split: str = "test",
        max_cases: int | None = None,
    ) -> int:
        output = Path(output_path)
        ensure_dir(output.parent)
        if output.exists():
            output.unlink()
        count = 0
        for idx, row in enumerate(self.iter_rows(dataset, input_paths, hf_name, split, max_cases)):
            case = self.build_case(row, index=idx)
            append_jsonl(output, case.model_dump())
            count += 1
        return count

    def pcu_response_records(self) -> list[dict[str, Any]]:
        return list(getattr(self.pcu_engine, "response_records", []))


def normalize_case(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages") or []
    first_user = ""
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                first_user = str(msg.get("content") or "")
                break
    problem = first_nonempty(
        [
            row.get("problem_statement"),
            row.get("problem"),
            row.get("issue"),
            row.get("issue_text"),
            row.get("description"),
            row.get("pr_description"),
            row.get("prompt"),
            extract_pr_description(first_user),
        ]
    )
    instance_id = first_nonempty([row.get("instance_id"), row.get("sample_name"), row.get("id")], default=stable_id("case", problem))
    repo = row.get("repo") or infer_repo_from_instance_id(instance_id)
    patch = first_nonempty([row.get("patch"), row.get("gold_patch"), row.get("reference_patch"), row.get("solution_patch")])
    test_patch = first_nonempty([row.get("test_patch"), row.get("tests"), row.get("test_diff"), row.get("fail_to_pass")])
    trajectory_info = extract_trajectory_info(messages)
    return {
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": row.get("base_commit") or row.get("commit") or "",
        "problem_statement": problem,
        "hints_text": first_nonempty([row.get("hints_text"), row.get("hints"), row.get("discussion")]),
        "created_at": row.get("created_at") or row.get("created"),
        "version": row.get("version"),
        "environment_setup_commit": row.get("environment_setup_commit"),
        "patch": patch,
        "test_patch": test_patch,
        "success": row.get("success"),
        "messages": messages,
        "logs": trajectory_info["logs"],
        "snippets": trajectory_info["snippets"],
        "detected_paths": trajectory_info["paths"],
        "observed_final_patch": first_nonempty([patch, trajectory_info["final_patch"]]),
        "_source_file": row.get("_source_file"),
        "_source_kind": "trajectory_jsonl" if messages else "swe_style_case",
    }


def infer_repo_from_instance_id(instance_id: str) -> str:
    if "__" not in instance_id:
        return ""
    left, right = instance_id.split("__", 1)
    repo_name = right.rsplit("-", 1)[0] if "-" in right else right
    if left and repo_name:
        return f"{left}/{repo_name}"
    return ""


def extract_trajectory_info(messages: list[dict[str, Any]] | Any) -> dict[str, Any]:
    if not isinstance(messages, list):
        return {"logs": [], "snippets": [], "paths": [], "final_patch": ""}
    logs: list[str] = []
    snippets: list[str] = []
    paths: list[str] = []
    final_patch = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "")
        paths.extend(path for path in extract_paths(content, limit=40) if path not in paths)
        if "diff --git " in content:
            final_patch = content[content.find("diff --git ") :]
        if msg.get("role") in {"tool", "user"} or msg.get("message_type") == "observation":
            low = content.lower()
            if looks_like_code_listing(content):
                snippets.append(truncate_text(content, 5000))
            elif looks_like_failure_log(content):
                logs.append(truncate_text(content, 4000))
    if not logs:
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                snippets.append(truncate_text(str(msg.get("content") or ""), 2400))
                if len(snippets) >= 2:
                    break
    return {"logs": logs[:3], "snippets": snippets[:3], "paths": paths[:25], "final_patch": final_patch}


def looks_like_code_listing(content: str) -> bool:
    if "Here's the result of running `cat -n`" in content:
        return True
    numbered = len(re.findall(r"(?m)^\s*\d+\t", content))
    return numbered >= 8


def looks_like_failure_log(content: str) -> bool:
    low = content.lower()
    strong = ["traceback", "assertionerror", "failed tests", " failures", " errors", "pytest", "failed:"]
    if any(marker in low for marker in strong):
        return True
    return bool(re.search(r"(?m)^(error|failed|failure):", content, re.I))


def build_context_sources(case: dict[str, Any]) -> list[ContextSource]:
    sources: list[ContextSource] = []

    def add(source: str, text: str, visible: bool = True, metadata: dict[str, Any] | None = None) -> None:
        if not text or not text.strip():
            return
        ref = stable_id(source, case["instance_id"], len(sources), text[:160])
        sources.append(
            ContextSource(
                ref_id=ref,
                source=source,
                text=text,
                token_count=approx_tokens(text),
                visible_to_agent=visible,
                metadata=metadata or {},
            )
        )

    add("issue", case.get("problem_statement", ""))
    add("discussion", case.get("hints_text", ""))
    for i, log in enumerate(case.get("logs", []) or []):
        add(
            "logs",
            log,
            metadata={
                "log_index": i,
                "diagnostic_confidence": "trajectory_low" if case.get("_source_kind") == "trajectory_jsonl" else "dataset_high",
            },
        )
    add("tests", case.get("test_patch", ""))

    detected_paths = case.get("detected_paths", []) or []
    if detected_paths:
        pseudo = "\n".join(f"- {path}" for path in detected_paths[:12])
        add(
            "snippets",
            "Plausible file paths observed during prior trajectories. Some may be distractors:\n" + pseudo,
            metadata={"kind": "candidate_paths", "may_include_distractors": True},
        )
    for i, snippet in enumerate(case.get("snippets", []) or []):
        add(
            "snippets",
            snippet,
            metadata={"kind": "trajectory_code_or_tool_observation", "snippet_index": i},
        )

    patch = case.get("patch") or case.get("observed_final_patch") or ""
    add("patch", patch, visible=False, metadata={"oracle_only": True, "files": patch_files(patch)})
    return sources


def build_interaction_script(
    case: dict[str, Any],
    sources: list[ContextSource],
    pcus: list[Any],
    track: TrackName,
    seed: int,
    noise_turns: int | None = None,
) -> list[InteractionTurn]:
    rnd = random.Random(seed)
    issue = next((s for s in sources if s.source == "issue"), None)
    logs = [s for s in sources if s.source == "logs"]
    tests = [s for s in sources if s.source == "tests"]
    snippets = [s for s in sources if s.source == "snippets"]
    hard_issue_pcus = [p.pcu_id for p in pcus if p.necessity == "hard" and any(span.source == "issue" for span in p.source_spans)]
    log_pcus = [p.pcu_id for p in pcus if any(span.source == "logs" for span in p.source_spans)]
    test_pcus = [p.pcu_id for p in pcus if any(span.source == "tests" for span in p.source_spans)]
    target_turns = 12 if track == "long_horizon" else 8
    if noise_turns is not None:
        target_turns = max(6, min(12, noise_turns + 4))

    turns: list[InteractionTurn] = []
    turns.append(
        InteractionTurn(
            turn_id=1,
            role="user",
            content=f"Repository: {case.get('repo') or 'unknown'}\nIssue:\n{issue.text if issue else case.get('problem_statement', '')}",
            tags=["issue", "pcu_intro"],
            introduced_pcus=hard_issue_pcus[:2],
            visible_ref_ids=[issue.ref_id] if issue else [],
        )
    )

    candidate_paths = case.get("detected_paths", []) or ["src/core.py", "tests/test_regression.py", "utils/helpers.py"]
    noise_templates = [
        "A previous run mentioned `{path}`. Treat it as a candidate only; it may be unrelated.",
        "Style note from review: prefer a small local fix over broad refactoring unless tests require it.",
        "Unverified suggestion: this might be caused by caching, but confirm against the concrete failure signal.",
        "Background chatter: dependency warnings and formatting noise appeared in the run, but they were not proven causal.",
    ]
    conflict_value_a = "None"
    conflict_value_b = "an empty list"

    turn_id = 2
    while len(turns) < target_turns - 3:
        if track == "long_horizon" and len(turns) == 3:
            turns.append(
                InteractionTurn(
                    turn_id=turn_id,
                    role="user",
                    content=f"Conflicting note A: one comment says returning {conflict_value_a} is acceptable. This note is intentionally low confidence.",
                    tags=["conflict", "noise"],
                )
            )
            turn_id += 1
            continue
        template = rnd.choice(noise_templates)
        turns.append(
            InteractionTurn(
                turn_id=turn_id,
                role="user",
                content=template.format(path=rnd.choice(candidate_paths)),
                tags=["noise"],
                visible_ref_ids=[snippets[0].ref_id] if snippets and rnd.random() < 0.4 else [],
            )
        )
        turn_id += 1

    if logs:
        key_log = select_key_log_excerpt(logs[0].text)
        long_log = make_long_log(key_log, rnd, lines=80 if track == "long_horizon" else 35)
        turns.append(
            InteractionTurn(
                turn_id=turn_id,
                role="user",
                content="Long diagnostic log follows. The useful line is buried inside noise:\n" + long_log,
                tags=["long_log", "pcu_intro"],
                introduced_pcus=log_pcus[:2],
                visible_ref_ids=[logs[0].ref_id],
            )
        )
        turn_id += 1

    if track == "long_horizon":
        turns.append(
            InteractionTurn(
                turn_id=turn_id,
                role="user",
                content=f"Conflicting note B: the later test-oriented clue says the behavior should produce {conflict_value_b} when the edge case is empty. Prefer stronger issue/test/log evidence over note A.",
                tags=["conflict_resolution", "delayed_dependency"],
                introduced_pcus=test_pcus[:1],
                visible_ref_ids=[tests[0].ref_id] if tests else [],
            )
        )
        turn_id += 1

    final_instruction = {
        "context_management": "Return only a JSON context_plan with keep, summarize, discard, and memory.",
        "budgeted_patch": "Return the current context_plan, any retrieval actions, and then prepare a patch under the token budget.",
        "long_horizon": "Before patching, update memory for delayed PCUs and resolve conflicts using issue/test/log evidence.",
    }[track]
    turns.append(
        InteractionTurn(
            turn_id=turn_id,
            role="evaluator",
            content=final_instruction,
            tags=["checkpoint", "requires_context_plan"],
        )
    )

    for idx, turn in enumerate(turns, start=1):
        turn.turn_id = idx
    return turns[:12]


def select_key_log_excerpt(log_text: str) -> str:
    for line in log_text.splitlines():
        if re.search(r"(traceback|assertionerror|exception|error:|failed|failure|expected|actual)", line, re.I):
            return line.strip()
    return truncate_text(log_text.strip(), 500)


def make_long_log(key_line: str, rnd: random.Random, lines: int) -> str:
    filler = [
        "INFO build cache hit for optional dependency",
        "DEBUG resolving package metadata",
        "WARNING unrelated deprecation in test harness",
        "INFO collected auxiliary files",
        "DEBUG retrying non-critical filesystem stat",
    ]
    insert_at = max(2, lines // 2)
    out = []
    for i in range(lines):
        if i == insert_at:
            out.append("CRITICAL " + key_line)
        else:
            out.append(f"{i:04d} {rnd.choice(filler)}")
    return "\n".join(out)
