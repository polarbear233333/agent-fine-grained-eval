from __future__ import annotations

from typing import Iterator, Dict, Any
from .trajectory_schema import Case
from .utils import read_jsonl


def load_cases(cfg: dict) -> Iterator[Case]:
    source = cfg["dataset"].get("source", "local_jsonl")
    max_cases = cfg.get("run", {}).get("max_cases")
    count = 0

    if source == "local_jsonl":
        for row in read_jsonl(cfg["dataset"]["local_path"]):
            yield Case(**normalize_case(row))
            count += 1
            if max_cases and count >= max_cases:
                break
        return

    if source == "swebench":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Please install datasets: pip install datasets") from exc
        ds = load_dataset(cfg["dataset"].get("hf_name", "princeton-nlp/SWE-bench_Lite"), split=cfg["dataset"].get("split", "test"))
        for row in ds:
            yield Case(**normalize_case(dict(row)))
            count += 1
            if max_cases and count >= max_cases:
                break
        return

    raise ValueError(f"Unsupported dataset.source: {source}")


def normalize_case(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "instance_id": row.get("instance_id") or row.get("id"),
        "repo": row.get("repo"),
        "base_commit": row.get("base_commit"),
        "problem_statement": row.get("problem_statement") or row.get("issue") or row.get("prompt") or "",
        "hints_text": row.get("hints_text") or row.get("hints") or "",
        "patch": row.get("patch"),
        "test_patch": row.get("test_patch"),
    }
