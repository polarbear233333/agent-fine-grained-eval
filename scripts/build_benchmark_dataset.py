from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_eval.dataset_builder import DatasetBuilder  # noqa: E402
from agent_eval.llm_pcu_engine import PCU_PROMPT_VERSION  # noqa: E402
from agent_eval.manifest import write_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PCU Context Benchmark dataset JSONL.")
    parser.add_argument("--track", default="A")
    parser.add_argument("--dataset", default="local")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--max-visible-tokens", type=int, default=8192)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--pcu-mode", default="heuristic", choices=["heuristic", "llm", "hybrid"])
    parser.add_argument("--pcu-model", default="gpt-5.4-mini")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    inputs = [part.strip() for value in args.input for part in value.split(",") if part.strip()]
    builder = DatasetBuilder(
        track=args.track,
        max_visible_tokens=args.max_visible_tokens,
        memory_slots=args.memory_slots,
        pcu_mode=args.pcu_mode,
        pcu_model=args.pcu_model,
        seed=args.seed,
    )
    count = builder.build_file(args.dataset, args.output, input_paths=inputs, max_cases=args.max_cases)
    manifest_path = Path(str(args.output) + ".manifest.json")
    write_manifest(
        manifest_path,
        command="python scripts/build_benchmark_dataset.py",
        config=vars(args),
        artifacts=[{"name": "converted_dataset", "path": args.output}],
        api_calls=builder.pcu_response_records(),
        prompt_versions={"pcu": PCU_PROMPT_VERSION},
    )
    print(f"Wrote {count} cases to {args.output}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
