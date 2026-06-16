# Real SWE Dataset Conversion Summary

This repository now supports direct conversion from Hugging Face SWE-bench and SWE-Pro datasets.

## Data Sources

| alias | Hugging Face dataset | split tested |
|---|---|---|
| `swebench_verified` | `SWE-bench/SWE-bench_Verified` | `test` |
| `swepro` | `ScaleAI/SWE-bench_Pro` | `test` |

## Conversion Commands

```bash
python scripts/convert_real_swe.py \
  --dataset swebench_verified \
  --split test \
  --track A \
  --output experiments/real_swe_verified_10_hybrid.jsonl \
  --max-cases 10 \
  --pcu-mode hybrid \
  --pcu-model gpt-5.4-mini
```

```bash
python scripts/convert_real_swe.py \
  --dataset swepro \
  --split test \
  --track A \
  --output experiments/real_swepro_10_hybrid.jsonl \
  --max-cases 10 \
  --pcu-mode hybrid \
  --pcu-model gpt-5.4-mini
```

## Outputs

| output | cases | avg turns | avg PCUs | hard PCUs | soft PCUs | PCU sources |
|---|---:|---:|---:|---:|---:|---|
| `real_swe_verified_10_hybrid.jsonl` | 10 | 6.0 | 6.0 | 36 | 24 | issue/tests/patch/discussion |
| `real_swepro_10_hybrid.jsonl` | 10 | 6.0 | 5.4 | 33 | 21 | issue/tests/patch |
| `real_swepro_100_hybrid.jsonl` | 100 | 6.0 | 6.25 | 366 | 259 | issue/tests/patch |

## SWE-Pro 100 Manifest

`experiments/real_swepro_100_hybrid.jsonl.manifest.json` records:

- dataset alias: `swepro`
- Hugging Face dataset: `ScaleAI/SWE-bench_Pro`
- split: `test`
- PCU mode: `hybrid`
- PCU model: `gpt-5.4-mini`
- prompt version: `llm-pcu-v1`
- converted dataset SHA-256: `02955716fed267dba14080855b3771a071392888756c9ecf9f4b740525e3bbba`
- API call records: 101

## SWE-Pro 100 Ablation

Proxy Full Context vs Masked PCU ablation:

```bash
python scripts/run_ablation.py \
  --dataset experiments/real_swepro_100_hybrid.jsonl \
  --output-dir experiments/ablation_swepro_100_proxy \
  --solver-mode proxy \
  --necessity hard \
  --max-pcus-per-case 2
```

Result:

| output | ablations | annotated hard | inferred hard | mean success delta | necessity agreement |
|---|---:|---:|---:|---:|---:|
| `ablation_swepro_100_proxy` | 192 | 192 | 192 | 0.55 | 1.0 |

## Example PCU Differences

Astropy Verified case:

- nested compound models must preserve correct separability information
- expected separability matrix should remain block-diagonal
- test patch encodes expected 4D matrix behavior

NodeBB SWE-Pro case:

- ACP user data must expose validated/pending/expired/missing email state
- confirmation validity depends on explicit expiry metadata
- validation actions must recover email from alternate validation storage

These examples show that PCUs are not fixed templates. They are generated per case by the LLM annotator and then grounded back to source spans.
