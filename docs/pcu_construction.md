# PCU Construction Pipeline

PCU construction is case-specific. The system does not assign a fixed list of PCUs to every case. Each case is first normalized into structured context sources, and then PCUs are extracted from the semantics of that case.

## Inputs

Each SWE-style case is normalized into:

```json
{
  "instance_id": "...",
  "repo": "...",
  "base_commit": "...",
  "context_sources": [
    {"ref_id": "issue-...", "source": "issue", "text": "..."},
    {"ref_id": "tests-...", "source": "tests", "text": "..."},
    {"ref_id": "patch-...", "source": "patch", "text": "...", "visible_to_agent": false}
  ]
}
```

For existing agent trajectories, tool observations are split into:

- `logs`: high-signal failure-like outputs
- `snippets`: code listings or noisy tool observations
- `patch`: oracle/final patch if available

Trajectory logs are treated as lower-confidence than dataset-provided tests or issue text.

## Modes

### Heuristic

Command:

```bash
python scripts/convert_real_swe.py --dataset swebench_verified --pcu-mode heuristic ...
```

The heuristic extractor uses conservative rules:

- issue sentences containing `should`, `expected`, `actual`, `bug`, `regression`, etc.
- test lines containing assertions, regression tests, or expected outputs
- failure-log lines containing traceback/error/failure signals
- gold patch summaries as soft oracle PCUs

This mode is fast and reproducible, but it should be treated as a bootstrap baseline.

### LLM

Command:

```bash
python scripts/convert_real_swe.py --dataset swebench_verified --pcu-mode llm --pcu-model gpt-5.4-mini ...
```

The LLM receives the full per-case context package and returns:

```json
{
  "pcus": [
    {
      "necessity": "hard",
      "source": "issue",
      "ref_id": "issue-...",
      "evidence_quote": "exact substring copied from source text",
      "description": "why this information is causal",
      "expected_patch_effect": "semantic repair behavior",
      "importance": 0.9
    }
  ]
}
```

The model does not provide trusted offsets. Instead, it provides an `evidence_quote`. The script maps that quote back into the original source text and computes exact `start`/`end` offsets. This gives span-level PCUs while avoiding hallucinated offsets.

### Hybrid

Command:

```bash
python scripts/convert_real_swe.py --dataset swepro --pcu-mode hybrid --pcu-model gpt-5.4-mini ...
```

Hybrid mode uses LLM PCUs as the primary case-specific annotation and adds non-duplicate patch-oracle soft PCUs from the heuristic engine. This is the recommended mode for benchmark construction before human review.

## Why This Is Case-Specific

For an Astropy separability bug, hard PCUs may involve nested `CompoundModel` semantics and expected block-diagonal matrices.

For a NodeBB email validation issue, hard PCUs may involve ACP email state, fallback email lookup, and confirmation-expiration semantics.

The PCU list is therefore determined by each case's issue, tests, logs, and patch semantics. The only shared code is the annotation protocol and grounding mechanism.

## Current Limitations

- LLM PCUs are still annotations, not causal proof.
- Hard/soft labels should be calibrated with Direct Solve ablation.
- Some cases need human adjudication when issue text and test patch imply different repair scopes.
- Patch-derived PCUs are oracle-only and must not be exposed to the agent during Track A/B/C runs.

## Recommended Paper-Grade Workflow

1. Generate candidate PCUs with `--pcu-mode hybrid`.
2. Human-review a stratified subset.
3. Run Direct Solve ablation:
   - Full Context
   - Masked PCU span
   - compare patch/test success
4. Calibrate hard/soft thresholds.
5. Freeze a PCU-gold JSONL for final benchmark release.

