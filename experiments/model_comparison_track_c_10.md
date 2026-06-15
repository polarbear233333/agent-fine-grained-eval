# Track C Long-Horizon Stress Test

Dataset: first 10 cases from `data/glm-think-claude_1_it1_final_data.json`.

Track: C / long-horizon interference. Runs include delayed PCU checkpoints, conflict hints, long logs, and distractor paths. Local patch execution was disabled, so `task_success` is not applicable.

| model | cases | hard PCU recall | soft PCU recall | PCU Recall@K | delayed recall | CBR | retrieval cost | patch alignment proxy | conflict resolution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heuristic | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 19.0790 | 0.8260 | 0.3500 | 0.9000 |
| gpt-5.3-codex-spark | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 83.8809 | 1.3843 | 0.2500 | 0.9000 |
| gpt-5.5 | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 53.7823 | 3.3395 | 0.3000 | 1.0000 |

Notes:

- All runs retain hard and soft PCUs on this small stress subset.
- `gpt-5.5` has better conflict-resolution accuracy and lower CBR than `gpt-5.3-codex-spark`, but higher retrieval cost.
- These Track C numbers should be treated as a smoke/stress sample. Scale to 50+ cases after fixing the final model matrix and budget.

