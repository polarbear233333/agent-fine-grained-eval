from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_eval.runner import parse_args, run_experiment  # noqa: E402


def main() -> None:
    args = parse_args()
    run_dir = run_experiment(args)
    print(f"Experiment complete: {run_dir}")


if __name__ == "__main__":
    main()

