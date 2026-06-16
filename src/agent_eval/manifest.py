from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import file_sha256, write_json


SCHEMA_VERSION = "pcu-bench-manifest-v1"


def current_git_commit(cwd: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_manifest(
    path: str | Path,
    *,
    command: str,
    config: dict[str, Any],
    artifacts: list[dict[str, Any]],
    api_calls: list[dict[str, Any]] | None = None,
    prompt_versions: dict[str, str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config": config,
        "git_commit": current_git_commit("."),
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
        },
        "prompt_versions": prompt_versions or {},
        "artifacts": with_checksums(artifacts),
        "api_calls": api_calls or [],
        "notes": notes or [],
    }
    write_json(path, manifest)
    return manifest


def with_checksums(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for artifact in artifacts:
        item = dict(artifact)
        path = item.get("path")
        if path and Path(path).exists() and Path(path).is_file():
            item.setdefault("sha256", file_sha256(path))
            item.setdefault("bytes", Path(path).stat().st_size)
        out.append(item)
    return out
