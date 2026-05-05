from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from .utils import ensure_dir, write_json


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_reports(task_decomp_report: str | Path | None, context_report: str | Path | None, output_dir: str | Path) -> Dict[str, Any]:
    out = ensure_dir(output_dir)
    td_raw = load_json(task_decomp_report) if task_decomp_report and Path(task_decomp_report).exists() else []
    ce_raw = load_json(context_report) if context_report and Path(context_report).exists() else []
    td: List[Dict[str, Any]] = td_raw.get("cases", []) if isinstance(td_raw, dict) else td_raw
    ce: List[Dict[str, Any]] = ce_raw.get("cases", []) if isinstance(ce_raw, dict) else ce_raw
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in td:
        by_id.setdefault(item["instance_id"], {"instance_id": item["instance_id"]})["task_decomposition"] = item.get("metrics", {})
    for item in ce:
        by_id.setdefault(item["instance_id"], {"instance_id": item["instance_id"]})["context_engineering"] = item.get("metrics", {})
    rows = []
    for instance_id, item in by_id.items():
        td_score = item.get("task_decomposition", {}).get("final_score", 0.0)
        ce_score = item.get("context_engineering", {}).get("final_ce_score", 0.0)
        item["integrated_score"] = round(0.5 * td_score + 0.5 * ce_score, 4)
        rows.append(item)
    summary = {
        "case_count": len(rows),
        "avg_integrated_score": round(sum(r["integrated_score"] for r in rows) / max(1, len(rows)), 4),
        "cases": rows,
    }
    write_json(out / "integrated_capability_report.json", summary)
    return summary
