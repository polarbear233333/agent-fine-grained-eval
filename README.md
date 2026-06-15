# PCU Context Engineering Benchmark

This repository is a research harness for evaluating context engineering in software-engineering agents. It extends a SWE-bench-style patch-and-test workflow with explicit measurement of:

- Context Budgeting: what the agent keeps, summarizes, or discards under a visible-token budget.
- Information Retention: whether task-critical facts survive long multi-turn interaction.
- Actionable Recall: whether retained facts causally affect the final patch or repair decision.
- Task Decomposition: whether the agent plans and executes a reasonable engineering workflow.

The current implementation is designed as a first paper-grade pipeline: modular, reproducible, JSONL-native, and safe to run in offline smoke mode before attaching costly model or SWE-bench execution backends.

## Design Background

SWE-bench evaluates whether a model can resolve real GitHub issues by generating patches against pinned repositories. The public SWE-bench site reports the family of benchmark splits, including Full, Lite, Verified, Multilingual, and Multimodal; Verified is a 500-instance human-filtered subset. The official SWE-bench repository documents the patch evaluation harness and its Docker-based reproducibility requirements.

SWE-bench Pro pushes toward longer-horizon, more realistic agent tasks. Public descriptions report 1,865 problems across 41 repositories, with public, held-out, and commercial subsets, and emphasize multi-file, enterprise-like tasks that remain substantially harder than SWE-bench Verified.

This project keeps the SWE-bench idea of "issue + repository + patch + tests", but adds a benchmark layer for context decisions and memory. It can ingest SWE-style JSONL, Hugging Face SWE-bench datasets, and existing agent trajectory JSONL.

References:

- SWE-bench website: https://www.swebench.com/
- SWE-bench GitHub: https://github.com/SWE-bench/SWE-bench
- SWE-bench Pro public dataset page: https://labs.scale.com/leaderboard/swe_bench_pro_public
- SWE-bench Pro repository: https://github.com/scaleapi/SWE-bench_Pro-os

## Core Concept: Patch Causal Unit

A Patch Causal Unit (PCU) is a minimal causal information unit:

> If the agent does not know or use this information, it is unlikely to generate a correct patch that passes the task tests.

PCUs can be extracted from:

- issue constraints and acceptance criteria
- failure logs and key test errors
- tests or regression assertions
- gold patch semantics
- long interaction logs containing buried task triggers

The required public PCU shape is:

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
  "expected_patch_effect": "The semantic repair behavior affected by this information."
}
```

The internal schema also stores optional `ref_id`, `description`, `importance`, and `ablation` metadata so experiments can trace a PCU back to exact context sources.

### Necessity

- Hard PCU: missing the unit should sharply reduce patch success.
- Soft PCU: missing the unit may still pass tests but makes the repair weaker, less robust, or less aligned with edge cases.

The PCU engine exposes direct ablation:

1. Full Context: solve with all context sources.
2. Masked Context: remove the candidate PCU span.
3. Compare pass rates.
4. Use the success delta to classify hard vs. soft.

Offline dataset construction uses a deterministic proxy ablation so the pipeline is reproducible without burning model budget. Replace the solver callback in `PCUEngine.validate_with_ablation()` for final paper experiments.

## Modules

The benchmark is split into the requested research modules.

### `dataset_builder`

File: `src/agent_eval/dataset_builder.py`

Converts SWE-bench/SWE-Pro/local trajectory cases into benchmark instances containing:

- `context_sources`
- `interaction_script`
- `pcus`
- `context_budget`

It builds 6-12 turn scripts with:

- noise suggestions
- conflicting low-confidence hints
- long diagnostic logs with buried signals
- pseudo-related paths and snippets
- delayed PCU checkpoints

It supports:

- local SWE-style JSONL
- existing trajectory JSONL with `sample_name`, `messages`, and `success`
- Hugging Face SWE-bench family datasets when `datasets` is installed

### `pcu_engine`

File: `src/agent_eval/pcu_engine.py`

Extracts PCUs from issue/log/test/patch sources and records span-level evidence. It implements:

- issue constraint extraction
- log/test signal extraction
- gold patch semantic extraction
- hard/soft classification
- offline proxy ablation
- real direct-solve ablation hook

Trajectory observations are treated conservatively: logs produced by a prior agent run are low-confidence unless direct ablation or stronger task evidence validates them.

### `context_manager`

File: `src/agent_eval/context_manager.py`

Simulates realistic agent memory:

- `max_visible_tokens`
- `memory_slots`
- `keep`
- `summarize`
- `discard`
- memory eviction
- forgetting events
- budget overflow accounting

Agent output format:

```json
{
  "keep": [],
  "summarize": [],
  "discard": [],
  "memory": []
}
```

The manager executes the plan, updates the memory buffer, and returns the visible context state used for scoring.

### `swe_bench_runner`

File: `src/agent_eval/swe_bench_runner.py`

Provides a minimal local execution boundary:

- clone repo
- checkout commit
- apply patch
- run tests
- collect pass/fail, logs, and diff
- remove temporary worktree by default

Track B/C default to `--runner-mode skip` so smoke runs do not clone large repos or leave worktrees. Use `--runner-mode local` only when the repository, commit, dependencies, and test command are ready.

### `trajectory_logger`

File: `src/agent_eval/trajectory_logger.py`

Writes JSONL events:

- `case_start`
- `agent_output`
- `context_state`
- `swe_runner`
- `metrics`
- `task_decomposition`

The canonical output is:

```text
experiments/
  run_xxx/
    config.json
    converted_dataset.jsonl
    dataset_analysis.json
    trajectories.jsonl
    metrics.json
    summary.md
```

### `evaluation_engine`

File: `src/agent_eval/evaluation_engine.py`

Implements benchmark metrics:

- Task Success
- PCU Recall@K
- Hard PCU Recall
- Soft PCU Recall
- PCU-to-Patch Alignment
- Context Bloat Ratio
- Retrieval Cost
- Delayed PCU Recall
- Forgetting Events
- Conflict Resolution Accuracy

Metric definitions:

```text
Task Success = 1 if tests pass, else 0
PCU Recall@K = covered top-K PCUs / K
Hard PCU Recall = covered hard PCUs / total hard PCUs
Soft PCU Recall = covered soft PCUs / total soft PCUs
CBR = retained visible tokens / minimal tokens covering hard PCU spans
Retrieval Cost = sum(action cost + retrieved token cost)
Delayed PCU Recall = recall of PCUs introduced before a delayed checkpoint
Forgetting Events = PCUs removed from memory after being stored
Conflict Resolution Accuracy = whether stronger PCU evidence overrides conflicting noise
PCU-to-Patch Alignment = hard PCUs whose expected patch effect appears in the final diff
```

### `task_decomposition_eval`

File: `src/agent_eval/task_decomposition_eval.py`

Wraps the existing `td_pipeline` slice extractor and heuristic judge. It implements the paper rubric:

- 0: no planning
- 1: weak planning
- 2: partial planning
- 3: standard engineering planning
- 4: systematic planning
- 5: advanced dynamic planning

Aggregate formulas:

```text
PQS = average slice score
PAR = aligned plan steps / executed steps
PRQ = reasonable plan revisions / total revisions
EE = execution efficiency
Final TD Score = 0.4 * PQS_normalized + 0.3 * PAR + 0.2 * PRQ + 0.1 * EE
```

## Tracks

### Track A: Pure Context Management

The agent only returns `context_plan`. No patch is required.

Use this to isolate budgeting, retention, and memory behavior.

### Track B: SWE-bench + Token Budget

The agent must manage context and produce a patch. If `--runner-mode local` is enabled, the runner applies the patch and executes tests.

Use this to test whether context management improves actual task success.

### Track C: Long-Horizon Interference

Track B plus stronger long-horizon stressors:

- delayed PCUs
- conflicting hints
- long logs
- distractor files
- memory stability checks

Use this to study forgetting and conflict resolution.

## Installation

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

For Hugging Face SWE-bench loading:

```bash
pip install -e ".[swebench]"
```

For tests:

```bash
pip install -e ".[dev]"
pytest
```

## CLI

Required entry point:

```bash
python run.py --track A --model gpt-4.1 --dataset swebench
```

Offline local trajectory smoke run:

```bash
python run.py ^
  --track A ^
  --model heuristic ^
  --dataset trajectories ^
  --input data\glm-think-claude_1_it1_final_data.json ^
  --max-cases 50 ^
  --run-name local_50_track_a
```

Long-horizon dataset construction only:

```bash
python scripts/build_benchmark_dataset.py ^
  --track C ^
  --dataset trajectories ^
  --input data\glm-think-claude_1_it1_final_data.json ^
  --output experiments\track_c_dataset.jsonl ^
  --max-cases 50
```

Track B with local patch execution:

```bash
python run.py ^
  --track B ^
  --model gpt-4.1 ^
  --agent-provider openai_compatible ^
  --dataset swebench_verified ^
  --max-cases 10 ^
  --runner-mode local ^
  --test-command "python -m pytest"
```

The OpenAI-compatible agent uses `OPENAI_API_KEY` or `JUDGE_API_KEY`. For custom routers, set `OPENAI_BASE_URL`, `API_BASE`, or `JUDGE_BASE_URL`; they are checked in that order.

## Data Format

SWE-style input:

```json
{
  "instance_id": "django__django-11119",
  "repo": "django/django",
  "base_commit": "abc123",
  "problem_statement": "Bug report...",
  "hints_text": "Optional discussion...",
  "patch": "diff --git ...",
  "test_patch": "diff --git ..."
}
```

Trajectory input:

```json
{
  "sample_name": "django__django-11119",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<pr_description>...</pr_description>"},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "content": "..."}
  ],
  "success": true
}
```

Converted benchmark instance:

```json
{
  "benchmark_id": "pcu-context-bench-v1",
  "instance_id": "repo__issue",
  "track": "context_management",
  "metadata": {},
  "task": {},
  "context_sources": [],
  "interaction_script": [],
  "pcus": [],
  "context_budget": {},
  "evaluation": {}
}
```

## Reproducibility

Every run writes:

- `config.json`: exact CLI arguments
- `converted_dataset.jsonl`: benchmark cases generated from the source data
- `dataset_analysis.json`: source and PCU counts
- `trajectories.jsonl`: event-level agent and evaluator trace
- `metrics.json`: per-case and aggregate scores
- `summary.md`: human-readable run summary

Use `--seed` to make noise injection and interaction scripts deterministic.

Raw trajectory files in `data/` are ignored by Git because they can be hundreds of MB. Keep them in local storage or publish them separately with a dataset artifact.

## Current Status

Implemented:

- modular PCU benchmark package under `src/agent_eval`
- PCU extraction and ablation interface
- context manager with token and memory constraints
- Track A/B/C dataset construction
- heuristic offline agent
- optional OpenAI-compatible context-plan agent
- local SWE runner boundary with cleanup
- JSONL trajectory logging
- PCU/context metrics
- task decomposition evaluator wrapper
- CLI and dataset builder script
- smoke tests

Remaining for final paper experiments:

- replace offline proxy ablation with model-based Direct Solve ablations
- run Dockerized SWE-bench harness for high-confidence Task Success
- calibrate PCU labels with human review on a subset
- report noise-level curves and confidence intervals
- attach a fixed model/scaffold matrix for Track B/C
