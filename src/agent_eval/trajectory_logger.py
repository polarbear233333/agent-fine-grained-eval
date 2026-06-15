from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import append_jsonl, ensure_dir


class TrajectoryLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        if self.path.exists():
            self.path.unlink()

    def log(self, event_type: str, instance_id: str, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.path,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event_type,
                "instance_id": instance_id,
                **payload,
            },
        )

