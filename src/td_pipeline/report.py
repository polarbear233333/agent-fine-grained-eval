from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict, Any
from .utils import write_json


def write_case_score(path: str | Path, obj: Dict[str, Any]) -> None:
    write_json(path, obj)


def write_final_reports(output_dir: str | Path, case_scores: List[Dict[str, Any]]) -> None:
    output_dir = Path(output_dir)
    write_json(output_dir / "final_report.json", {"case_count": len(case_scores), "cases": case_scores})
    csv_path = output_dir / "final_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["instance_id", "slice_count", "pqs", "par", "prq", "ee", "final_score", "final_score_5pt"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for cs in case_scores:
            m = cs.get("metrics", {})
            writer.writerow({
                "instance_id": cs.get("instance_id"),
                "slice_count": cs.get("slice_count", 0),
                "pqs": m.get("pqs"),
                "par": m.get("par"),
                "prq": m.get("prq"),
                "ee": m.get("ee"),
                "final_score": m.get("final_score"),
                "final_score_5pt": m.get("final_score_5pt"),
            })
