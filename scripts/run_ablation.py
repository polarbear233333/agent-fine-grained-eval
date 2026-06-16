from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_eval.ablation_runner import AblationRunner, summarize_ablation_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Full Context vs Masked PCU ablations.")
    parser.add_argument("--dataset", required=True, help="Converted benchmark JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--solver-mode", default="proxy", choices=["proxy", "llm_proxy"])
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-pcus-per-case", type=int, default=None)
    parser.add_argument("--necessity", default="all", choices=["all", "hard", "soft"])
    parser.add_argument("--include-oracle-patch-pcus", action="store_true")
    args = parser.parse_args()

    runner = AblationRunner(solver_mode=args.solver_mode, model=args.model)
    results = runner.run_file(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        max_pcus_per_case=args.max_pcus_per_case,
        necessity=args.necessity,
        include_oracle_patch_pcus=args.include_oracle_patch_pcus,
    )
    print(f"Wrote {len(results)} ablations to {args.output_dir}")
    print(summarize_ablation_results(results))


if __name__ == "__main__":
    main()

