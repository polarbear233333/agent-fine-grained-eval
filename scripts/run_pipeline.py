from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm
from td_pipeline.config import load_config
from td_pipeline.swebench_loader import load_cases
from td_pipeline.sii_client import SIIClient
from td_pipeline.slice_extractor import extract_slices
from td_pipeline.judge_client import JudgeClient
from td_pipeline.aggregator import aggregate_scores
from td_pipeline.report import write_case_score, write_final_reports
from td_pipeline.utils import ensure_dir, write_json, write_jsonl, safe_filename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rubric = cfg["rubric_def"]
    out = ensure_dir(cfg["run"].get("output_dir", "outputs/task_decomp_run"))
    traj_dir = ensure_dir(out / "trajectories")
    slice_dir = ensure_dir(out / "slices")
    score_dir = ensure_dir(out / "scores")

    cases = list(load_cases(cfg))
    write_jsonl(out / "cases.jsonl", [c.model_dump() for c in cases])

    sii = SIIClient(cfg)
    judge = JudgeClient(cfg)
    case_scores = []

    for case in tqdm(cases, desc="Evaluating cases"):
        name = safe_filename(case.instance_id)
        traj = sii.run_case(case)
        write_json(traj_dir / f"{name}.trajectory.json", traj.model_dump())

        slices = extract_slices(traj, rubric)
        write_json(slice_dir / f"{name}.slices.json", [s.model_dump() for s in slices])

        slice_outputs = []
        judge_scores = []
        for s in slices:
            judge_result = judge.score_slice(case.problem_statement, s.model_dump(), rubric)
            judge_scores.append(judge_result)
            item = s.model_dump()
            item["judge"] = judge_result
            slice_outputs.append(item)

        metrics = aggregate_scores(judge_scores, rubric)
        score_obj = {
            "instance_id": case.instance_id,
            "slice_count": len(slices),
            "metrics": metrics,
            "slices": slice_outputs,
        }
        write_case_score(score_dir / f"{name}.score.json", score_obj)
        case_scores.append(score_obj)

    write_final_reports(out, case_scores)
    print(f"Done. Reports saved to: {out}")


if __name__ == "__main__":
    main()
