from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm

from .ce_schema import ContextEngineeringInstance
from .config import load_config
from .context_judge import ContextJudgeClient
from .utils import read_jsonl, write_json, write_jsonl, ensure_dir, safe_filename


def mock_agent_context_plan(instance: ContextEngineeringInstance) -> Dict[str, Any]:
    hard_refs = []
    mem = []
    for pcu in instance.pcus:
        if pcu.necessity == "hard":
            hard_refs.extend([s.ref_id for s in pcu.source_spans])
            mem.append({"key": pcu.pcu_id, "value": pcu.description[:240]})
    summarize = []
    for src in instance.context_sources:
        if src.ref_id not in hard_refs and src.source in {"discussion", "logs", "tests"}:
            summarize.append({"ref": src.ref_id, "summary": src.text[:180].replace("\n", " ")})
    discard = [src.ref_id for src in instance.context_sources if src.source in {"patch", "metadata"}]
    return {
        "context_plan": {
            "keep": hard_refs,
            "summarize": summarize[:3],
            "discard": discard,
            "memory": mem[: instance.context_budget.memory_slots],
        },
        "actions": [{"type": "external_retrieve", "query": "search files related to hard PCUs"}],
        "final_patch": "diff --git a/example.py b/example.py\n# placeholder patch for CE scoring demo",
    }


def run_context_eval(dataset_path: str | Path, cfg: dict, output_dir: str | Path) -> List[Dict[str, Any]]:
    out = ensure_dir(output_dir)
    inst_dir = ensure_dir(out / "ce_instances")
    score_dir = ensure_dir(out / "ce_scores")
    judge = ContextJudgeClient(cfg)
    scores = []
    for row in tqdm(read_jsonl(dataset_path), desc="Evaluating context engineering"):
        inst = ContextEngineeringInstance(**row)
        agent_output = mock_agent_context_plan(inst)
        score = judge.score(inst, agent_output)
        obj = {"instance_id": inst.instance_id, "track": inst.track, "agent_output": agent_output, "metrics": score}
        name = safe_filename(inst.instance_id)
        write_json(inst_dir / f"{name}.ce_instance.json", inst.model_dump())
        write_json(score_dir / f"{name}.ce_score.json", obj)
        scores.append(obj)
    write_json(out / "context_engineering_report.json", scores)
    write_jsonl(out / "context_engineering_report.jsonl", scores)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Context Engineering evaluation on converted CE JSONL.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, help="Converted CE JSONL")
    parser.add_argument("--output-dir", default="outputs/context_engineering_run")
    args = parser.parse_args()
    cfg = load_config(args.config)
    scores = run_context_eval(args.dataset, cfg, args.output_dir)
    avg = sum(s["metrics"]["final_ce_score"] for s in scores) / max(1, len(scores))
    print(f"Done. Context Engineering cases={len(scores)}, avg_final_ce_score={avg:.4f}, output={args.output_dir}")


if __name__ == "__main__":
    main()
