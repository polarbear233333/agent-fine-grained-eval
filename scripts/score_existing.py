from __future__ import annotations

import argparse
from td_pipeline.config import load_config
from td_pipeline.trajectory_schema import Trajectory, Case
from td_pipeline.slice_extractor import extract_slices
from td_pipeline.judge_client import JudgeClient
from td_pipeline.aggregator import aggregate_scores
from td_pipeline.utils import load_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--issue-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rubric = cfg["rubric_def"]
    traj = Trajectory(**load_json(args.trajectory))
    cases = [Case(**row) for row in read_jsonl(args.issue_file)]
    case = next((c for c in cases if c.instance_id == traj.instance_id), cases[0])

    slices = extract_slices(traj, rubric)
    judge = JudgeClient(cfg)
    outputs = []
    scores = []
    for s in slices:
        jr = judge.score_slice(case.problem_statement, s.model_dump(), rubric)
        scores.append(jr)
        item = s.model_dump()
        item["judge"] = jr
        outputs.append(item)

    result = {
        "instance_id": traj.instance_id,
        "slice_count": len(slices),
        "metrics": aggregate_scores(scores, rubric),
        "slices": outputs,
    }
    write_json(args.output, result)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
