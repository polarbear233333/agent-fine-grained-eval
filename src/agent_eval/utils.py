from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_filename(value: str, limit: int = 180) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in value)
    return safe[:limit] or "unnamed"


def stable_id(prefix: str, *parts: object, length: int = 10) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return f"{prefix}-{h.hexdigest()[:length]}"


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latinish = max(0, len(text) - cjk)
    return max(1, cjk + latinish // 4)


def read_jsonl_stream(path: str | Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                break


def write_json(path: str | Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_pr_description(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"<pr_description>\s*(.*?)\s*</pr_description>", text, re.S | re.I)
    if match:
        return match.group(1).strip()
    match = re.search(r"Consider the following PR description:\s*(.*)", text, re.S | re.I)
    if match:
        return match.group(1).strip()
    return text.strip()


def iter_sentences_with_offsets(text: str) -> Iterator[tuple[str, int, int]]:
    if not text:
        return
    pattern = re.compile(r"[^.!?\n\r]+(?:[.!?]+|\n+|$)", re.M)
    for match in pattern.finditer(text):
        sentence = match.group(0).strip()
        if sentence:
            yield sentence, match.start(), match.end()


def first_nonempty(values: Sequence[Any], default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text
    return default


def truncate_text(text: str, max_chars: int, suffix: str = "\n[truncated]") -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(suffix))
    return text[:keep].rstrip() + suffix


def extract_paths(text: str, limit: int = 20) -> list[str]:
    if not text:
        return []
    pattern = re.compile(r"(?:/testbed/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.(?:py|js|ts|tsx|java|go|rs|rb|php|c|cc|cpp|h|hpp)")
    seen: list[str] = []
    for match in pattern.finditer(text):
        path = match.group(0).replace("/testbed/", "")
        if path not in seen:
            seen.append(path)
        if len(seen) >= limit:
            break
    return seen


def patch_files(patch: str, limit: int = 20) -> list[str]:
    files: list[str] = []
    if not patch:
        return files
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                path = parts[3].removeprefix("b/")
                if path not in files:
                    files.append(path)
        elif line.startswith("+++ b/"):
            path = line.removeprefix("+++ b/")
            if path != "/dev/null" and path not in files:
                files.append(path)
        if len(files) >= limit:
            break
    return files


def added_patch_lines(patch: str, limit: int = 80) -> list[str]:
    lines: list[str] = []
    for line in (patch or "").splitlines():
        if line.startswith("+") and not line.startswith("+++") and line.strip("+").strip():
            lines.append(line[1:])
        if len(lines) >= limit:
            break
    return lines


def mask_spans(text: str, spans: Sequence[tuple[int, int]], marker: str = "[MASKED_PCU]") -> str:
    if not spans:
        return text
    normalized = sorted((max(0, s), min(len(text), e)) for s, e in spans if e > s)
    out: list[str] = []
    cursor = 0
    for start, end in normalized:
        if start < cursor:
            continue
        out.append(text[cursor:start])
        out.append(marker)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def keyword_set(text: str, max_words: int = 24) -> set[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "should",
        "would",
        "could",
        "patch",
        "issue",
        "problem",
        "expected",
        "actual",
    }
    result: list[str] = []
    for word in words:
        if word in stop:
            continue
        if word not in result:
            result.append(word)
        if len(result) >= max_words:
            break
    return set(result)

