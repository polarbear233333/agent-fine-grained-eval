from __future__ import annotations

import json

SYSTEM_PROMPT = """You are an expert software engineering evaluator.
Your goal is to evaluate the Task Decomposition capability of an autonomous coding agent.
You must be strict, evidence-based, and return valid JSON only.
"""


def build_judge_prompt(issue: str, td_slice: dict, rubric: dict) -> str:
    labels = rubric.get("score_labels", {})
    return f"""
Evaluate the following Task Decomposition Slice.

Issue Description:
{issue}

Planning Thought:
{td_slice.get('planning_text', '')}

Subsequent Actions and Observations:
{td_slice.get('execution_text', '')}

Plan Revision Evidence:
{td_slice.get('revision_text', '')}

Rubric:
Score 0: No planning. The agent performs random actions without a clear plan.
Score 1: Weak planning. The agent expresses a vague idea but does not decompose the task.
Score 2: Partial decomposition. Some steps are mentioned but the plan is incomplete.
Score 3: Reasonable decomposition. The agent provides a clear plan including reproduction, localization, modification, and verification.
Score 4: Systematic decomposition. The agent demonstrates structured planning and follows the plan effectively.
Score 5: Advanced decomposition. The agent dynamically revises and optimizes the plan based on new information.

Score labels:
{json.dumps(labels, ensure_ascii=False, indent=2)}

Return JSON only with this exact schema:
{{
  "score": 0,
  "classification_label": "...",
  "reasoning": "Brief evidence-based explanation.",
  "covered_stages": {{
    "reproduction": true,
    "localization": true,
    "modification": true,
    "verification": true
  }},
  "aligned_steps": 0,
  "total_executed_steps": 0,
  "reasonable_revisions": 0,
  "total_revisions": 0,
  "efficiency_score": 0.0,
  "evidence": ["short quote or paraphrase 1", "short quote or paraphrase 2"]
}}

Rules:
- score must be an integer from 0 to 5.
- efficiency_score must be between 0 and 1.
- aligned_steps means executed steps that clearly correspond to the stated plan.
- total_executed_steps means meaningful actions, not passive thoughts.
- reasonable_revisions counts revisions that are justified by observations.
- total_revisions counts all detected plan changes.
- Do not reward long trajectories merely because they contain many actions.
""".strip()
