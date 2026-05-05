from __future__ import annotations

from typing import List, Dict, Any


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return default if b == 0 else a / b


def aggregate_scores(slice_scores: List[Dict[str, Any]], rubric: dict) -> Dict[str, float]:
    if not slice_scores:
        return {
            "pqs": 0.0,
            "pqs_normalized": 0.0,
            "par": 0.0,
            "prq": 0.0,
            "ee": 0.0,
            "final_score": 0.0,
            "final_score_5pt": 0.0,
        }
    pqs = sum(s["score"] for s in slice_scores) / len(slice_scores)
    aligned = sum(s.get("aligned_steps", 0) for s in slice_scores)
    total_steps = sum(s.get("total_executed_steps", 0) for s in slice_scores)
    reasonable_revisions = sum(s.get("reasonable_revisions", 0) for s in slice_scores)
    total_revisions = sum(s.get("total_revisions", 0) for s in slice_scores)
    ee = sum(s.get("efficiency_score", 0.0) for s in slice_scores) / len(slice_scores)

    par = safe_div(aligned, total_steps, default=0.0)
    prq = safe_div(reasonable_revisions, total_revisions, default=1.0)
    pqs_norm = pqs / 5.0
    weights = rubric.get("aggregation_weights", {"pqs": 0.4, "par": 0.3, "prq": 0.2, "ee": 0.1})
    final = (
        weights.get("pqs", 0.4) * pqs_norm +
        weights.get("par", 0.3) * par +
        weights.get("prq", 0.2) * prq +
        weights.get("ee", 0.1) * ee
    )
    return {
        "pqs": round(pqs, 4),
        "pqs_normalized": round(pqs_norm, 4),
        "par": round(par, 4),
        "prq": round(prq, 4),
        "ee": round(ee, 4),
        "final_score": round(final, 4),
        "final_score_5pt": round(final * 5, 4),
    }
