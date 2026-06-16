# Full Context vs Masked PCU Ablation Runner

The ablation runner estimates PCU importance by comparing two input conditions:

1. Full Context: all visible context sources are provided unchanged.
2. Masked Context: the target PCU span is replaced with `[MASKED_PCU]`.

The runner writes one record per `(case, pcu)` pair:

```json
{
  "instance_id": "...",
  "pcu_id": "...",
  "annotated_necessity": "hard",
  "inferred_necessity": "hard",
  "full": {"success_score": 0.95},
  "masked": {"success_score": 0.40},
  "success_delta": 0.55
}
```

## Modes

### `proxy`

Cheap deterministic visibility proxy. It does not call a model and does not run tests. Use it for full-dataset smoke runs and checking whether masking logic works.

```bash
python scripts/run_ablation.py \
  --dataset experiments/real_swepro_100_hybrid.jsonl \
  --output-dir experiments/ablation_swepro_100_proxy \
  --solver-mode proxy \
  --necessity hard \
  --max-pcus-per-case 2
```

### `llm_proxy`

Calls a model twice per PCU: once with Full Context and once with Masked Context. The model proposes a semantic patch plan, and the runner scores whether the plan recovers the PCU's expected patch effect.

This is closer to Direct Solve than the deterministic proxy, but it is still not a SWE test-pass measurement.

```bash
python scripts/run_ablation.py \
  --dataset experiments/real_swepro_100_hybrid.jsonl \
  --output-dir experiments/ablation_swepro_llm_proxy_smoke \
  --solver-mode llm_proxy \
  --model gpt-5.4-mini \
  --max-cases 2 \
  --max-pcus-per-case 1 \
  --necessity hard
```

### Future `patch_test`

The intended paper-grade mode is:

```text
Full Context  -> model generates patch -> SWE tests
Masked PCU    -> model generates patch -> SWE tests
success_delta = pass_rate(full) - pass_rate(masked)
```

This needs Dockerized SWE-bench execution and should reuse `swe_bench_runner` or a full official SWE-bench harness integration.

## Oracle Patch PCUs

Patch-derived PCUs are skipped by default because gold patch context is marked `visible_to_agent=false`. Ablating a source that the agent never sees would not measure context retention.

Use `--include-oracle-patch-pcus` only for annotation analysis, not agent-facing experiments.

## Manifests

Every ablation run writes:

```text
artifact_manifest.json
```

It records:

- input dataset checksum
- output checksum
- prompt version
- model name
- API response ids for `llm_proxy`
- configuration
- git commit

