"""Preference-pair dataset emission.

Builds chosen/rejected pairs from the same task: the passing trajectory
is chosen, a failing trajectory on the same task is rejected. This is
the DPO/RLHF-ready format: {"prompt", "chosen", "rejected", "task_id"}.

Only tasks with at least one pass and at least one failure produce pairs.
"""

from __future__ import annotations

import json
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collector.trajectory import Trajectory


def build_pairs(trajectories: list[Trajectory]) -> list[dict]:
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        if t.verification is not None:
            by_task[t.task_id].append(t)

    pairs = []
    for task_id, group in by_task.items():
        passes = [t for t in group if t.verification.passed]
        fails = [t for t in group if not t.verification.passed]
        if not passes or not fails:
            continue
        # Best pass (highest pass rate, then shortest) vs worst fail.
        passes.sort(key=lambda t: (-t.verification.pass_rate, len(t.steps)))
        fails.sort(key=lambda t: (t.verification.pass_rate, -len(t.steps)))
        chosen, rejected = passes[0], fails[0]
        pairs.append({
            "prompt": chosen.prompt,
            "chosen": chosen.final_code,
            "rejected": rejected.final_code,
            "task_id": task_id,
            "task_type": chosen.task_type,
            "chosen_pass_rate": chosen.verification.pass_rate,
            "rejected_pass_rate": rejected.verification.pass_rate,
        })
    return pairs


def save_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
