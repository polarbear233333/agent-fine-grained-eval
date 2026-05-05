from __future__ import annotations
import argparse
from td_pipeline.integrated_report import merge_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-decomp-report", default="outputs/task_decomp_run/final_report.json")
    parser.add_argument("--context-report", default="outputs/context_engineering_run/context_engineering_report.json")
    parser.add_argument("--output-dir", default="outputs/integrated_run")
    args = parser.parse_args()
    summary = merge_reports(args.task_decomp_report, args.context_report, args.output_dir)
    print(f"Merged {summary['case_count']} cases. avg_integrated_score={summary['avg_integrated_score']}")


if __name__ == "__main__":
    main()
