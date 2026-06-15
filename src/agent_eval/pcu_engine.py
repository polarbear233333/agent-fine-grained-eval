from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .schema import AblationEvidence, ContextSource, PCU, SourceSpan
from .utils import added_patch_lines, approx_tokens, iter_sentences_with_offsets, keyword_set, patch_files, stable_id


SolverFn = Callable[[dict[str, str]], bool]


ISSUE_HARD_PATTERNS = re.compile(
    r"\b(should|must|expected|actual|fails?|failure|error|exception|traceback|bug|regression|incorrect|wrong|when|if)\b",
    re.I,
)
LOG_HARD_PATTERNS = re.compile(r"(traceback|assertionerror|exception|error:|failed|failure|expected|actual)", re.I)
TEST_SIGNAL_PATTERNS = re.compile(r"(\bassert\b|\bdef test_|pytest|unittest|expected|raises|xfail|regression)", re.I)


@dataclass
class CandidateSpan:
    source: ContextSource
    start: int
    end: int
    text: str
    necessity: str
    expected_effect: str
    description: str
    base_importance: float


class PCUEngine:
    """Extract and validate Patch Causal Units.

    The extraction is intentionally conservative. It produces span-level PCU
    candidates from issue text, failure logs, tests, and oracle gold patches.
    The ablation API can be backed by a real solve-and-test callback; when no
    callback is available it uses a deterministic proxy so dataset construction
    remains reproducible offline.
    """

    def extract_pcus(self, context_sources: list[ContextSource], metadata: dict[str, Any] | None = None) -> list[PCU]:
        metadata = metadata or {}
        candidates: list[CandidateSpan] = []
        for source in context_sources:
            if source.source == "issue":
                candidates.extend(self._issue_candidates(source))
            elif source.source in {"logs", "log"}:
                candidates.extend(self._log_candidates(source))
            elif source.source in {"tests", "test"}:
                candidates.extend(self._test_candidates(source))
            elif source.source in {"patch", "gold_patch"}:
                candidates.extend(self._patch_candidates(source, metadata))

        if not candidates:
            issue = next((s for s in context_sources if s.source == "issue"), None)
            if issue:
                text = issue.text[: min(700, len(issue.text))]
                candidates.append(
                    CandidateSpan(
                        source=issue,
                        start=0,
                        end=len(text),
                        text=text,
                        necessity="hard",
                        expected_effect="The patch must implement the main behavior requested by the issue.",
                        description="Fallback issue-level PCU because no finer-grained signal was detected.",
                        base_importance=0.7,
                    )
                )

        pcus: list[PCU] = []
        used_ids: set[str] = set()
        for idx, cand in enumerate(candidates[:8], start=1):
            pcu_id = stable_id("PCU", cand.source.ref_id, cand.start, cand.end, cand.text[:80], length=8)
            if pcu_id in used_ids:
                pcu_id = f"{pcu_id}-{idx}"
            used_ids.add(pcu_id)
            pcu = PCU(
                pcu_id=pcu_id,
                necessity="hard" if cand.necessity == "hard" else "soft",
                source_spans=[
                    SourceSpan(
                        source=cand.source.source,
                        ref_id=cand.source.ref_id,
                        start=cand.start,
                        end=cand.end,
                    )
                ],
                expected_patch_effect=cand.expected_effect,
                description=cand.description,
                importance=cand.base_importance,
            )
            pcu.ablation = self.proxy_ablation(pcu)
            if pcu.ablation.success_delta >= 0.35:
                pcu.necessity = "hard"
            elif pcu.ablation.success_delta <= 0.15:
                pcu.necessity = "soft"
            pcus.append(pcu)

        pcus.sort(key=lambda p: (p.necessity != "hard", -p.importance, p.pcu_id))
        return pcus

    def _issue_candidates(self, source: ContextSource) -> list[CandidateSpan]:
        candidates: list[CandidateSpan] = []
        for sentence, start, end in iter_sentences_with_offsets(source.text):
            if ISSUE_HARD_PATTERNS.search(sentence):
                candidates.append(
                    CandidateSpan(
                        source=source,
                        start=start,
                        end=end,
                        text=sentence,
                        necessity="hard",
                        expected_effect=f"Preserve the issue constraint: {sentence[:220]}",
                        description=f"Issue acceptance criterion or trigger condition: {sentence[:260]}",
                        base_importance=0.82 if any(w in sentence.lower() for w in ["should", "must", "expected"]) else 0.68,
                    )
                )
            if len(candidates) >= 2:
                break
        if not candidates and source.text.strip():
            end = min(len(source.text), 650)
            candidates.append(
                CandidateSpan(
                    source=source,
                    start=0,
                    end=end,
                    text=source.text[:end],
                    necessity="hard",
                    expected_effect="Implement the central behavior requested in the issue.",
                    description="Issue summary PCU.",
                    base_importance=0.7,
                )
            )
        return candidates

    def _log_candidates(self, source: ContextSource) -> list[CandidateSpan]:
        candidates: list[CandidateSpan] = []
        low_confidence = source.metadata.get("diagnostic_confidence") == "trajectory_low"
        for match in re.finditer(r"[^\n\r]*(?:traceback|assertionerror|exception|error:|failed|failure|expected|actual)[^\n\r]*", source.text, re.I):
            line = match.group(0).strip()
            if not line:
                continue
            candidates.append(
                CandidateSpan(
                    source=source,
                    start=match.start(),
                    end=match.end(),
                    text=line,
                    necessity="soft" if low_confidence else "hard",
                    expected_effect=f"Use or verify this diagnostic signal only if it matches the issue/test oracle: {line[:220]}"
                    if low_confidence
                    else f"Fix the behavior that causes this failure signal: {line[:220]}",
                    description=f"Low-confidence trajectory diagnostic signal: {line[:260]}"
                    if low_confidence
                    else f"Core failure/log signal: {line[:260]}",
                    base_importance=0.55 if low_confidence else 0.86,
                )
            )
            if len(candidates) >= 2:
                break
        return candidates

    def _test_candidates(self, source: ContextSource) -> list[CandidateSpan]:
        candidates: list[CandidateSpan] = []
        for match in re.finditer(r"[^\n\r]*(?:assert|def test_|pytest|expected|raises|regression)[^\n\r]*", source.text, re.I):
            line = match.group(0).strip()
            if not line:
                continue
            candidates.append(
                CandidateSpan(
                    source=source,
                    start=match.start(),
                    end=match.end(),
                    text=line,
                    necessity="hard" if TEST_SIGNAL_PATTERNS.search(line) else "soft",
                    expected_effect=f"Satisfy the test-level behavioral constraint: {line[:220]}",
                    description=f"Test oracle signal: {line[:260]}",
                    base_importance=0.78,
                )
            )
            if len(candidates) >= 2:
                break
        return candidates

    def _patch_candidates(self, source: ContextSource, metadata: dict[str, Any]) -> list[CandidateSpan]:
        files = patch_files(source.text, limit=5)
        added = added_patch_lines(source.text, limit=20)
        if not added and not files:
            return []
        summary_bits = []
        if files:
            summary_bits.append("files: " + ", ".join(files[:3]))
        if added:
            compact_added = " | ".join(line.strip() for line in added[:4] if line.strip())
            if compact_added:
                summary_bits.append("added semantics: " + compact_added[:240])
        summary = "; ".join(summary_bits)
        start = 0
        end = min(len(source.text), 1200)
        return [
            CandidateSpan(
                source=source,
                start=start,
                end=end,
                text=source.text[start:end],
                necessity="soft",
                expected_effect=f"Gold patch semantic target ({summary})",
                description=f"Oracle patch-derived semantic PCU for {metadata.get('instance_id', 'case')}: {summary}",
                base_importance=0.44,
            )
        ]

    def proxy_ablation(self, pcu: PCU) -> AblationEvidence:
        source_types = {span.source for span in pcu.source_spans}
        token_mass = sum(max(1, span.end - span.start) for span in pcu.source_spans)
        base = 0.92
        if source_types & {"issue", "tests"}:
            masked = 0.38 if pcu.importance >= 0.78 else 0.52
        elif source_types & {"logs"}:
            masked = 0.38 if pcu.importance >= 0.78 else 0.68
        elif source_types & {"patch", "gold_patch"}:
            masked = 0.74
        else:
            masked = 0.65
        if token_mass > 700:
            masked += 0.04
        masked = max(0.0, min(1.0, masked))
        return AblationEvidence(
            full_success_rate=base,
            masked_success_rate=masked,
            success_delta=round(base - masked, 4),
            trials=1,
            method="offline_proxy",
        )

    def validate_with_ablation(
        self,
        contexts_by_ref: dict[str, str],
        pcu: PCU,
        solver: SolverFn,
        trials: int = 1,
    ) -> AblationEvidence:
        """Run real Full Context vs Masked Context ablation when a solver exists.

        `solver` receives a mapping of ref_id to source text and returns whether
        the generated patch passed the tests. This boundary lets the benchmark
        plug in a costly Direct Solve backend without entangling PCU extraction
        with a specific model or SWE-bench harness.
        """
        full_success = 0
        masked_success = 0
        for _ in range(max(1, trials)):
            full_success += int(bool(solver(dict(contexts_by_ref))))
            masked_contexts = dict(contexts_by_ref)
            for span in pcu.source_spans:
                if not span.ref_id or span.ref_id not in masked_contexts:
                    continue
                text = masked_contexts[span.ref_id]
                masked_contexts[span.ref_id] = text[: span.start] + "[MASKED_PCU]" + text[span.end :]
            masked_success += int(bool(solver(masked_contexts)))
        total = max(1, trials)
        full_rate = full_success / total
        masked_rate = masked_success / total
        evidence = AblationEvidence(
            full_success_rate=round(full_rate, 4),
            masked_success_rate=round(masked_rate, 4),
            success_delta=round(full_rate - masked_rate, 4),
            trials=total,
            method="direct_solve",
        )
        pcu.ablation = evidence
        pcu.necessity = "hard" if evidence.success_delta >= 0.35 else "soft"
        pcu.importance = max(pcu.importance, evidence.success_delta)
        return evidence


def pcu_keywords(pcu: PCU) -> set[str]:
    return keyword_set(" ".join([pcu.pcu_id, pcu.description, pcu.expected_patch_effect]))


def pcu_minimal_tokens(pcus: list[PCU]) -> int:
    total_chars = 0
    for pcu in pcus:
        if pcu.necessity == "hard":
            total_chars += sum(max(1, span.end - span.start) for span in pcu.source_spans)
    return max(1, approx_tokens("x" * total_chars))
