from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

from .agents import make_agent
from .context_manager import ContextManager
from .dataset_builder import DatasetBuilder
from .evaluation_engine import EvaluationEngine
from .agents import AGENT_PROMPT_VERSION
from .llm_pcu_engine import PCU_PROMPT_VERSION
from .manifest import write_manifest
from .schema import CaseMetrics
from .swe_bench_runner import SWEBenchRunner
from .task_decomposition_eval import TaskDecompositionEvaluator
from .trajectory_logger import TrajectoryLogger
from .utils import append_jsonl, ensure_dir, safe_filename, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Context Engineering + PCU benchmark experiments.")
    parser.add_argument("--track", default="A", choices=["A", "B", "C", "context_management", "budgeted_patch", "long_horizon"])
    parser.add_argument("--model", default="heuristic", help="Model name recorded in outputs, e.g. gpt-4.1.")
    parser.add_argument("--agent-provider", default="heuristic", choices=["heuristic", "openai_compatible", "openai", "mock", "offline"])
    parser.add_argument("--dataset", default="local", help="local|jsonl|trajectories|swebench|swebench_verified|swepro")
    parser.add_argument("--input", action="append", default=[], help="Input JSONL path. Repeat for multiple local trajectory files.")
    parser.add_argument("--hf-name", default=None, help="Optional Hugging Face dataset name.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--max-visible-tokens", type=int, default=8192)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--noise-turns", type=int, default=None)
    parser.add_argument("--pcu-mode", default="heuristic", choices=["heuristic", "llm", "hybrid"], help="PCU construction mode.")
    parser.add_argument("--pcu-model", default="gpt-5.4-mini", help="Model used when --pcu-mode is llm or hybrid.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="experiments")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--runner-mode", default="skip", choices=["skip", "local"])
    parser.add_argument("--keep-worktree", action="store_true")
    parser.add_argument("--test-command", default=None)
    return parser.parse_args(argv)


def normalize_inputs(values: list[str]) -> list[str]:
    paths: list[str] = []
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if stripped:
                paths.append(stripped)
    return paths


def run_experiment(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(Path(args.output_root) / safe_filename(f"{run_name}_{args.track}_{args.model}"))
    dataset_path = run_dir / "converted_dataset.jsonl"
    if dataset_path.exists():
        dataset_path.unlink()

    config = vars(args).copy()
    write_json(run_dir / "config.json", config)

    builder = DatasetBuilder(
        track=args.track,
        max_visible_tokens=args.max_visible_tokens,
        memory_slots=args.memory_slots,
        noise_turns=args.noise_turns,
        seed=args.seed,
        pcu_mode=args.pcu_mode,
        pcu_model=args.pcu_model,
    )
    agent = make_agent(args.agent_provider, args.model)
    evaluator = EvaluationEngine()
    td_evaluator = TaskDecompositionEvaluator()
    logger = TrajectoryLogger(run_dir / "trajectories.jsonl")
    swe_runner = SWEBenchRunner(keep_worktree=args.keep_worktree)

    input_paths = normalize_inputs(args.input)
    metrics: list[CaseMetrics] = []
    agent_api_calls: list[dict[str, Any]] = []
    analysis = {
        "case_count": 0,
        "source_counts": {},
        "pcu_counts": {"hard": 0, "soft": 0},
        "trajectory_success_known": 0,
        "trajectory_success_rate": None,
    }
    known_success_values: list[bool] = []

    for idx, row in enumerate(
        builder.iter_rows(
            dataset=args.dataset,
            input_paths=input_paths,
            hf_name=args.hf_name,
            split=args.split,
            max_cases=args.max_cases,
        )
    ):
        case = builder.build_case(row, index=idx)
        append_jsonl(dataset_path, case.model_dump())
        update_analysis(analysis, case)
        if isinstance(case.metadata.get("trajectory_success"), bool):
            known_success_values.append(bool(case.metadata["trajectory_success"]))

        logger.log(
            "case_start",
            case.instance_id,
            {
                "track": case.track,
                "context_budget": case.context_budget.model_dump(),
                "pcu_count": len(case.pcus),
                "hard_pcus": [p.pcu_id for p in case.pcus if p.necessity == "hard"],
            },
        )
        if args.build_only:
            continue

        agent_output = agent.run(case)
        meta = agent_output.get("_meta") if isinstance(agent_output, dict) else None
        if isinstance(meta, dict):
            agent_api_calls.append(
                {
                    "kind": "agent_context_plan",
                    "model": meta.get("model"),
                    "response_id": meta.get("response_id"),
                    "prompt_version": meta.get("prompt_version"),
                    "instance_id": case.instance_id,
                    "attempt": meta.get("attempt"),
                }
            )
        logger.log("agent_output", case.instance_id, {"agent_output": agent_output})

        manager = ContextManager(case)
        state = manager.apply_plan(agent_output.get("context_plan", agent_output), turn_id=len(case.interaction_script))
        logger.log("context_state", case.instance_id, {"context_state": state.model_dump()})

        runner_result = None
        task_success = None
        final_patch = str(agent_output.get("final_patch") or "")
        if case.track in {"budgeted_patch", "long_horizon"}:
            if args.runner_mode == "local":
                runner_result = swe_runner.run(case, final_patch, test_command=args.test_command)
                task_success = runner_result.status == "pass"
                logger.log("swe_runner", case.instance_id, {"runner_result": runner_result.model_dump()})

        case_metrics = evaluator.score_case(
            case=case,
            agent_output=agent_output,
            state=state,
            task_success=task_success,
            final_patch=final_patch,
            actions=agent_output.get("actions", []),
        )
        metrics.append(case_metrics)
        logger.log("metrics", case.instance_id, {"metrics": case_metrics.model_dump()})

        messages = row.get("messages") if isinstance(row, dict) else None
        if isinstance(messages, list):
            td = td_evaluator.evaluate_messages(case.instance_id, messages)
            logger.log("task_decomposition", case.instance_id, td)

    if known_success_values:
        analysis["trajectory_success_known"] = len(known_success_values)
        analysis["trajectory_success_rate"] = round(sum(known_success_values) / len(known_success_values), 4)
    write_json(run_dir / "dataset_analysis.json", analysis)
    evaluator.write_outputs(str(run_dir), metrics, config)
    write_manifest(
        run_dir / "artifact_manifest.json",
        command="python run.py",
        config=config,
        artifacts=[
            {"name": "converted_dataset", "path": str(dataset_path)},
            {"name": "trajectories", "path": str(run_dir / "trajectories.jsonl")},
            {"name": "metrics", "path": str(run_dir / "metrics.json")},
            {"name": "dataset_analysis", "path": str(run_dir / "dataset_analysis.json")},
            {"name": "summary", "path": str(run_dir / "summary.md")},
        ],
        api_calls=builder.pcu_response_records() + agent_api_calls,
        prompt_versions={"pcu": PCU_PROMPT_VERSION, "agent": AGENT_PROMPT_VERSION},
    )
    return run_dir


def update_analysis(analysis: dict[str, Any], case: Any) -> None:
    analysis["case_count"] += 1
    for source in case.context_sources:
        analysis["source_counts"][source.source] = analysis["source_counts"].get(source.source, 0) + 1
    for pcu in case.pcus:
        analysis["pcu_counts"][pcu.necessity] = analysis["pcu_counts"].get(pcu.necessity, 0) + 1


def main() -> None:
    args = parse_args()
    run_dir = run_experiment(args)
    print(f"Experiment complete: {run_dir}")
