# PCU Context Engineering Benchmark

This repository is a research system for evaluating context engineering in software-engineering LLM agents. It builds on SWE-bench-style tasks, but the central object is not only whether the final patch passes tests. The benchmark also measures whether an agent can identify, preserve, compress, recall, and use the information that causally matters for the patch.

The current system supports:

- real SWE-bench Verified conversion
- real SWE-Pro conversion
- LLM-assisted Patch Causal Unit construction
- multi-turn context-stress scripts
- explicit memory and context-budget simulation
- JSONL trajectory logging
- cross-model Track A/C experiments
- Full Context vs Masked PCU ablation records
- artifact manifests with checksums, prompt versions, model ids, and API response ids

## Research Question

Modern coding agents fail not only because they cannot write code, but because they mishandle context. They keep noisy logs, forget early constraints, over-retain irrelevant files, or fail to connect a remembered fact to the final patch.

This benchmark studies three capabilities:

1. Context Budgeting: under a token budget, what does the agent keep, summarize, or discard?
2. Information Retention: do critical facts survive multi-turn interaction and interference?
3. Actionable Recall: does retained information affect the final repair behavior?

Task decomposition is evaluated as a complementary capability: whether the agent plans, localizes, modifies, verifies, and revises in a structured way.

## Patch Causal Unit

A Patch Causal Unit, or PCU, is a minimal information unit with causal relevance to the correct patch:

> If the agent does not know or use this information, it is unlikely to generate a patch equivalent to the gold repair.

Required public schema:

```json
{
  "pcu_id": "PCU-xxx",
  "necessity": "hard",
  "source_spans": [
    {
      "source": "issue",
      "start": 100,
      "end": 160
    }
  ],
  "expected_patch_effect": "The semantic repair behavior controlled by this information."
}
```

Internal records also keep `ref_id`, `description`, `importance`, and ablation metadata so every PCU can be traced to exact source text.

### Hard vs Soft

- Hard PCU: removing it should sharply reduce the chance of producing the correct patch.
- Soft PCU: useful for robustness, edge cases, style, or patch quality, but not always strictly necessary.

The repository now includes a first ablation runner for Full Context vs Masked PCU. Proxy modes are available today; the interface is designed so true Dockerized SWE pass/fail can be plugged in later.

## PCU Construction

PCUs are not fixed templates. They are generated per case.

For each SWE/SWE-Pro instance, the system first builds:

```text
context_sources:
  issue
  discussion
  logs
  tests
  snippets
  patch      # oracle-only; visible_to_agent=false
```

Then it supports three PCU modes:

```text
heuristic  deterministic bootstrap extractor
llm        model-generated case-specific PCUs
hybrid     LLM PCUs plus non-duplicate patch-oracle soft PCUs
```

In LLM and hybrid modes, the model receives the case context and returns PCUs with `evidence_quote`. The script then grounds each quote back to the original source text and computes exact character offsets. The model is not trusted to invent offsets.

Example difference:

- Astropy separability case: PCUs involve nested CompoundModels and expected separability matrices.
- NodeBB email validation case: PCUs involve ACP email status, fallback email lookup, and confirmation expiry semantics.

Detailed construction notes are in [docs/pcu_construction.md](docs/pcu_construction.md).
Full Context vs Masked PCU ablation details are in [docs/ablation_runner.md](docs/ablation_runner.md).

## Tracks

### Track A: Context Management

The agent only outputs a context plan:

```json
{
  "keep": [],
  "summarize": [],
  "discard": [],
  "memory": []
}
```

No patch is required. This isolates context budgeting and PCU retention.

### Track B: Budgeted Patch

The agent must manage context and produce a patch. If `--runner-mode local` is enabled, the patch is applied and tests are run. If runner mode is skipped, `task_success` remains `null`.

### Track C: Long-Horizon Interference

Track B plus:

- delayed PCUs
- conflicting hints
- long noisy logs
- distractor files
- memory stability checks

## Repository Layout

```text
src/agent_eval/
  schema.py                  benchmark, PCU, memory, metrics schemas
  dataset_builder.py         local/SWE/SWE-Pro to benchmark JSONL
  pcu_engine.py              heuristic PCU extraction and ablation hook
  llm_pcu_engine.py          LLM PCU annotation and quote grounding
  context_manager.py         keep/summarize/discard/memory execution
  evaluation_engine.py       PCU recall, CBR, delayed recall, alignment
  ablation_runner.py         Full Context vs Masked PCU runner
  swe_bench_runner.py        minimal clone/apply/test boundary
  task_decomposition_eval.py planning slice and PQS/PAR/PRQ/EE wrapper
  trajectory_logger.py       JSONL event logging
  manifest.py                checksums, prompt versions, response ids
  agents.py                  heuristic and OpenAI-compatible agents

scripts/
  build_benchmark_dataset.py
  convert_real_swe.py
  run_ablation.py
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev,swebench]"
```

The project also works without installation when run from the repository root because scripts add `src/` to `PYTHONPATH`.

For OpenAI-compatible model calls:

```powershell
$env:OPENAI_API_KEY="..."
$env:API_BASE="https://4router.net/v1"
```

Supported base URL variables are checked in this order:

```text
OPENAI_BASE_URL
API_BASE
JUDGE_BASE_URL
```

## Convert Real SWE Data

SWE-bench Verified:

```bash
python scripts/convert_real_swe.py ^
  --dataset swebench_verified ^
  --split test ^
  --track A ^
  --output experiments\real_swe_verified_10_hybrid.jsonl ^
  --max-cases 10 ^
  --pcu-mode hybrid ^
  --pcu-model gpt-5.4-mini
```

SWE-Pro:

```bash
python scripts/convert_real_swe.py ^
  --dataset swepro ^
  --split test ^
  --track A ^
  --output experiments\real_swepro_100_hybrid.jsonl ^
  --max-cases 100 ^
  --pcu-mode hybrid ^
  --pcu-model gpt-5.4-mini
```

Data aliases:

| alias | Hugging Face dataset |
|---|---|
| `swebench` | `SWE-bench/SWE-bench` |
| `swebench_lite` | `SWE-bench/SWE-bench_Lite` |
| `swebench_verified` | `SWE-bench/SWE-bench_Verified` |
| `swepro` | `ScaleAI/SWE-bench_Pro` |

Every conversion writes a sidecar manifest:

```text
<output>.manifest.json
```

The manifest records:

- dataset/config
- output checksum
- prompt version
- model used for PCU annotation
- API response ids
- artifact sizes
- git commit and platform

## Run Context Experiments

Track A with a model:

```bash
python run.py ^
  --track A ^
  --model gpt-5.5 ^
  --agent-provider openai_compatible ^
  --dataset trajectories ^
  --input data\glm-think-claude_1_it1_final_data.json ^
  --max-cases 50 ^
  --run-name router_50_track_a_gpt-5.5
```

Track C stress sample:

```bash
python run.py ^
  --track C ^
  --model gpt-5.5 ^
  --agent-provider openai_compatible ^
  --dataset trajectories ^
  --input data\glm-think-claude_1_it1_final_data.json ^
  --max-cases 10 ^
  --run-name track_c_10_gpt-5.5
```

Run outputs:

```text
experiments/run_xxx/
  config.json
  converted_dataset.jsonl
  dataset_analysis.json
  trajectories.jsonl
  metrics.json
  summary.md
  artifact_manifest.json
```

## Run PCU Ablations

Full Context vs Masked PCU proxy ablation:

```bash
python scripts/run_ablation.py ^
  --dataset experiments\real_swepro_100_hybrid.jsonl ^
  --output-dir experiments\ablation_swepro_100_proxy ^
  --solver-mode proxy ^
  --necessity hard ^
  --max-pcus-per-case 2
```

Small LLM proxy ablation:

```bash
python scripts/run_ablation.py ^
  --dataset experiments\real_swepro_100_hybrid.jsonl ^
  --output-dir experiments\ablation_swepro_llm_proxy_smoke ^
  --solver-mode llm_proxy ^
  --model gpt-5.4-mini ^
  --max-cases 2 ^
  --max-pcus-per-case 1 ^
  --necessity hard
```

Ablation outputs:

```text
ablations.jsonl
ablation_metrics.json
artifact_manifest.json
```

Important: `proxy` and `llm_proxy` are not official SWE test-pass measurements. They are infrastructure for estimating PCU importance before the Dockerized SWE harness is attached. Patch-only oracle PCUs are skipped by default because patch sources are not visible to agents.

## Metrics

Core metrics:

```text
Task Success = tests pass, only when runner-mode local is used
PCU Recall@K = covered top-K PCUs / K
Hard PCU Recall = covered hard PCUs / total hard PCUs
Soft PCU Recall = covered soft PCUs / total soft PCUs
PCU-to-Patch Alignment = hard PCUs whose semantic effect appears in final patch
CBR = retained visible tokens / minimal tokens covering hard PCUs
Retrieval Cost = sum(action cost + retrieved token cost)
Delayed PCU Recall = recall after delayed dependency checkpoints
Forgetting Events = PCUs removed from memory after being stored
Conflict Resolution Accuracy = whether stronger evidence overrides noise
```

Task decomposition metrics:

```text
PQS = average planning quality
PAR = aligned plan steps / executed steps
PRQ = reasonable revisions / total revisions
EE = execution efficiency
Final TD Score = 0.4 * PQS + 0.3 * PAR + 0.2 * PRQ + 0.1 * EE
```

## Current Artifacts

Existing experiment summaries:

- [experiments/model_comparison_track_a_50.md](experiments/model_comparison_track_a_50.md)
- [experiments/model_comparison_track_c_10.md](experiments/model_comparison_track_c_10.md)
- [experiments/real_dataset_conversion_summary.md](experiments/real_dataset_conversion_summary.md)

Real converted data samples:

- `experiments/real_swe_verified_10_hybrid.jsonl`
- `experiments/real_swepro_10_hybrid.jsonl`
- `experiments/real_swepro_100_hybrid.jsonl` when the 100-case expansion is run

## Scientific Status

This is now an LLM-assisted PCU benchmark construction pipeline, not just a heuristic demo. It can create case-specific PCUs for real SWE-bench and SWE-Pro tasks, ground them to spans, stress agents with long-horizon context scripts, and record reproducible manifests.

Remaining work for a paper-grade benchmark:

1. Human review and adjudication of LLM-generated PCUs.
2. Direct Solve ablation with real patch generation and SWE tests.
3. Dockerized SWE-bench harness integration for Track B/C.
4. Budget sensitivity curves over token limits and memory slots.
5. Confidence intervals and failure taxonomy.
6. Frozen PCU-gold release with checksums and annotation guidelines.

## Notes

Raw trajectory files under `data/` are ignored because they can be hundreds of MB. Publish them as separate dataset artifacts rather than committing them to Git.

If an API key has appeared in logs or chat, rotate it before publishing the repository.
