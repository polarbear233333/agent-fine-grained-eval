# SWE Agent Capability Eval

A project-level evaluation pipeline for SWE-bench-style autonomous coding agents. It currently integrates two complementary rubrics:

1. **Task Decomposition**: evaluates whether an agent can turn a GitHub issue into an executable engineering plan and follow the sequence `Reproduce → Localize → Modify → Verify`.
2. **Context Engineering**: evaluates whether an agent can keep, summarize, discard, recall, and apply critical information under a limited context budget.

The project is designed for lab/server use: it can load SWE-bench cases, send them to a deployed SII/coding-agent endpoint, collect trajectories, call an LLM judge, and export unified JSON/CSV reports. It also includes an offline heuristic mode so the whole repo can be demonstrated without API keys.

---

## 1. What this pipeline does

```text
SWE-bench / local cases
        |
        | 1. load or convert data
        v
SII / coding agent execution
        |
        | 2. collect trajectory JSON
        v
Task Decomposition evaluator
        |
        | 3. extract planning slices + score with rubric
        v
Context Engineering converter/evaluator
        |
        | 4. build PCUs, context budget, interaction script + score context_plan
        v
Unified capability report
```

### Task Decomposition module

The module extracts **Task Planning Slices** from the trajectory and scores each slice on a 0–5 scale:

- 0: no task decomposition
- 1: weak task decomposition
- 2: partial decomposition
- 3: reasonable decomposition
- 4: systematic decomposition
- 5: advanced planning with revision/adaptivity

It then aggregates:

```text
Final TD Score = 0.4 * PQS + 0.3 * PAR + 0.2 * PRQ + 0.1 * EE
```

where:

- `PQS`: planning quality score
- `PAR`: plan adherence rate
- `PRQ`: plan revision quality
- `EE`: execution efficiency

### Context Engineering module

The Context Engineering part converts SWE-bench-style cases into a benchmark format with:

- `context_sources`: issue, discussion, logs, tests, snippets, patch metadata
- `interaction_script`: multi-turn session with noise/distractions
- `pcus`: Patch Causal Units, i.e. task-critical information units
- `context_budget`: visible-token limit, memory slots, allowed operations

It evaluates:

- `hard_pcu_recall`
- `soft_pcu_recall`
- `context_bloat_ratio`
- `memory_utilization`
- `delayed_recall_accuracy`
- `conflict_resolution_accuracy`
- `noise_resistance_score`
- `forgetting_events_count`
- `actionable_recall_score`
- `final_ce_score`

---

## 2. Project structure

```text
swe-agent-capability-eval/
├── configs/
│   ├── default.yaml
│   ├── rubric.task_decomp.yaml
│   └── rubric.context_engineering.yaml
├── examples/
│   ├── sample_cases.jsonl
│   └── sample_trajectory.json
├── scripts/
│   ├── run_pipeline.py              # Task Decomposition full run
│   ├── score_existing.py            # Score an existing trajectory
│   ├── convert_context_dataset.py   # SWE-bench -> Context Engineering JSONL
│   ├── run_context_eval.py          # Context Engineering evaluation
│   └── merge_reports.py             # Merge TD + CE reports
├── src/td_pipeline/
│   ├── swebench_loader.py
│   ├── sii_client.py
│   ├── slice_extractor.py
│   ├── judge_client.py
│   ├── aggregator.py
│   ├── context_converter.py
│   ├── context_judge.py
│   ├── context_runner.py
│   ├── integrated_report.py
│   └── ...
├── tests/
├── pyproject.toml
└── README.md
```

---

## 3. Installation

```bash
git clone <your-repo-url>
cd swe-agent-capability-eval
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
```

For loading SWE-bench directly from Hugging Face:

```bash
pip install -e ".[swebench]"
```

For development tests:

```bash
pip install -e ".[dev]"
pytest
```

---

## 4. Run the offline demo

The default configuration uses:

```yaml
sii:
  mode: mock
judge:
  provider: heuristic
```

So it does not require a real SII server or LLM API.

### 4.1 Task Decomposition demo

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

Outputs:

```text
outputs/task_decomp_run/
├── cases.jsonl
├── trajectories/*.trajectory.json
├── slices/*.slices.json
├── scores/*.score.json
├── final_report.json
└── final_report.csv
```

### 4.2 Convert data for Context Engineering

```bash
python scripts/convert_context_dataset.py \
  --input examples/sample_cases.jsonl \
  --output outputs/context_dataset/ce_cases.jsonl \
  --track long_horizon \
  --max-visible-tokens 8192 \
  --memory-slots 8 \
  --noise-turns 8
```

This creates SWECE-style cases with PCUs, interaction scripts, and context budgets.

### 4.3 Run Context Engineering evaluation

```bash
python scripts/run_context_eval.py \
  --config configs/default.yaml \
  --dataset outputs/context_dataset/ce_cases.jsonl \
  --output-dir outputs/context_engineering_run
```

Outputs:

```text
outputs/context_engineering_run/
├── ce_instances/*.ce_instance.json
├── ce_scores/*.ce_score.json
├── context_engineering_report.json
└── context_engineering_report.jsonl
```

### 4.4 Merge reports

```bash
python scripts/merge_reports.py \
  --task-decomp-report outputs/task_decomp_run/final_report.json \
  --context-report outputs/context_engineering_run/context_engineering_report.json \
  --output-dir outputs/integrated_run
```

Output:

```text
outputs/integrated_run/integrated_capability_report.json
```

---

## 5. Use with a real SII server

Edit `configs/default.yaml`:

```yaml
sii:
  mode: http
  base_url: http://YOUR_SII_SERVER:PORT
  endpoint: /run
  timeout_sec: 600
```

Expected SII endpoint behavior:

```http
POST /run
Content-Type: application/json
```

Request body:

```json
{
  "instance_id": "...",
  "repo": "...",
  "base_commit": "...",
  "problem_statement": "...",
  "hints_text": "..."
}
```

Expected response body:

```json
{
  "instance_id": "...",
  "steps": [
    {
      "index": 0,
      "role": "assistant",
      "thought": "Plan: ...",
      "action": "run_tests",
      "observation": "..."
    }
  ],
  "final_patch": "diff --git ..."
}
```

The pipeline will save this as a trajectory and evaluate it.

---

## 6. Use with an LLM judge

Create `.env`:

```bash
cp .env.example .env
```

Fill in:

```bash
JUDGE_API_KEY=your_api_key
JUDGE_BASE_URL=https://your-openai-compatible-endpoint/v1
JUDGE_MODEL=your_model_name
```

Then edit `configs/default.yaml`:

```yaml
judge:
  provider: openai_compatible
  temperature: 0
  max_retries: 3
```

The judge client uses OpenAI-compatible chat completion APIs and requests strict JSON output.

---

## 7. Input data format

Local JSONL cases should follow SWE-bench-like fields:

```json
{
  "instance_id": "repo__issue-1",
  "repo": "owner/repo",
  "base_commit": "abc123",
  "problem_statement": "Bug report or issue description...",
  "hints_text": "Optional discussion or hints...",
  "patch": "Optional gold patch...",
  "test_patch": "Optional test patch..."
}
```

To use Hugging Face SWE-bench directly, edit:

```yaml
dataset:
  source: swebench
  hf_name: princeton-nlp/SWE-bench_Lite
  split: test
```

Then run:

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

---

## 8. Context Engineering converted format

Each converted item has this top-level schema:

```json
{
  "benchmark_id": "swece-v1",
  "instance_id": "repo__issue__hash",
  "track": "context_management | budgeted_patch | long_horizon",
  "metadata": {},
  "task": {},
  "context_sources": [],
  "interaction_script": [],
  "pcus": [],
  "context_budget": {},
  "evaluation": {}
}
```

A PCU looks like:

```json
{
  "pcu_id": "PCU-issue-core",
  "necessity": "hard",
  "description": "Core user-reported bug/request...",
  "source_spans": [
    {
      "source": "issue",
      "ref_id": "issue-xxxx",
      "start": 0,
      "end": 600
    }
  ],
  "expected_patch_effect": {
    "file": "path/to/file.py",
    "semantic_change": "The patch should address the required behavior."
  }
}
```

---

## 9. Recommended experiment workflow

For real experiments, use this order:

```bash
# 1. Run coding agent and task decomposition evaluation
python scripts/run_pipeline.py --config configs/default.yaml

# 2. Convert the same cases into Context Engineering benchmark cases
python scripts/convert_context_dataset.py \
  --input examples/sample_cases.jsonl \
  --output outputs/context_dataset/ce_cases.jsonl \
  --track long_horizon

# 3. Run CE evaluation
python scripts/run_context_eval.py \
  --config configs/default.yaml \
  --dataset outputs/context_dataset/ce_cases.jsonl

# 4. Merge reports
python scripts/merge_reports.py
```

---

## 10. Notes

- The heuristic judge is only for pipeline verification and GitHub demo.
- For paper/report experiments, use `judge.provider=openai_compatible`.
- PCU generation in this repo is a practical bootstrap implementation. For strict benchmark construction, PCUs should be refined with ablation validation or human review.
- Context Engineering and Task Decomposition are intentionally separated in data conversion, but merged at the final reporting layer.

---

## 11. License

MIT License. Replace this section with your lab/project license if needed.
