# Track A Model Comparison

Dataset: first 50 cases from `data/glm-think-claude_1_it1_final_data.json`.

Track: A / context management only. No patch execution was required, so `task_success` is not applicable.

| model | cases | hard PCU recall | soft PCU recall | PCU Recall@K | delayed recall | CBR | retrieval cost | patch alignment proxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heuristic | 50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 18.5778 | 0.0000 | 0.0000 |
| gpt-5.4-mini | 50 | 1.0000 | 0.9867 | 0.9920 | 0.9800 | 72.2436 | 1.0237 | 0.0300 |
| gpt-5.3-codex-spark | 50 | 1.0000 | 0.9853 | 0.9893 | 0.9800 | 71.0736 | 1.5609 | 0.1900 |
| gpt-5.4 | 50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 87.0607 | 2.4759 | 0.0100 |
| gpt-5.5 | 50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 78.4004 | 3.0093 | 0.2300 |

Notes:

- All model runs preserve hard PCUs on this Track A subset.
- The main separation is efficiency: model runs retain much more context than the heuristic baseline, producing high Context Bloat Ratio.
- `gpt-5.3-codex-spark` and `gpt-5.5` show higher patch-alignment proxy despite Track A not requiring patch generation.
- `gpt-5.5` initially produced one malformed JSON response; the agent client was updated with JSON extraction and retry logic, then the 50-case run completed successfully.

