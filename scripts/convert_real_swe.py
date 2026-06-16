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


DEFAULT_HF = {
    "swebench": "SWE-bench/SWE-bench",
    "swebench_lite": "SWE-bench/SWE-bench_Lite",
    "swebench_verified": "SWE-bench/SWE-bench_Verified",
    "swepro": "ScaleAI/SWE-bench_Pro",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert real SWE-bench/SWE-Pro datasets into PCU Context Benchmark JSONL.")
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULT_HF), help="Real dataset family to load from Hugging Face.")
    parser.add_argument("--hf-name", default=None, help="Override Hugging Face dataset name.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--track", default="A", choices=["A", "B", "C", "context_management", "budgeted_patch", "long_horizon"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--max-visible-tokens", type=int, default=8192)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--pcu-mode", default="heuristic", choices=["heuristic", "llm", "hybrid"])
    parser.add_argument("--pcu-model", default="gpt-5.4-mini")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    builder = DatasetBuilder(
        track=args.track,
        max_visible_tokens=args.max_visible_tokens,
        memory_slots=args.memory_slots,
        seed=args.seed,
        pcu_mode=args.pcu_mode,
        pcu_model=args.pcu_model,
    )
    count = builder.build_file(
        dataset=args.dataset,
        output_path=args.output,
        hf_name=args.hf_name or DEFAULT_HF[args.dataset],
        split=args.split,
        max_cases=args.max_cases,
    )
    manifest_path = Path(str(args.output) + ".manifest.json")
    write_manifest(
        manifest_path,
        command="python scripts/convert_real_swe.py",
        config=vars(args),
        artifacts=[{"name": "converted_dataset", "path": args.output}],
        api_calls=builder.pcu_response_records(),
        prompt_versions={"pcu": PCU_PROMPT_VERSION},
        notes=["Patch context sources are oracle-only and marked visible_to_agent=false."],
    )
    print(f"Wrote {count} {args.dataset} cases to {args.output} with pcu_mode={args.pcu_mode}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
