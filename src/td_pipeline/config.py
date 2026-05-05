from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> Dict[str, Any]:
    load_dotenv()
    cfg = load_yaml(path)
    rubric_path = cfg.get("rubric", {}).get("path")
    if rubric_path:
        cfg["rubric_def"] = load_yaml(rubric_path)
    return cfg
