"""Curriculum ordering: easy to hard.

Training on randomly shuffled data wastes the early steps of training
on examples the model cannot yet use. Curriculum learning orders the
dataset by difficulty so the model masters simple patterns before
facing hard ones.

Difficulty is estimated per trajectory from three signals, all
available without human labels:

- trajectory shape (0.35): longer trajectories with more tool calls
  took more work; normalized against a 20-step cap.
- task failure rate (0.35): the fraction of attempts at this task that
  failed. Tasks the agent usually fails are hard tasks.
- code complexity (0.30): branch count from the AST, normalized.

Rows are emitted in curriculum order with `difficulty` and
`curriculum_order` attached, and can be binned into phases for staged
training (warm up on phase 1, finish on phase N).
"""

from __future__ import annotations

import ast
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collector.trajectory import Trajectory
from datasets.sft import to_sft_row

MAX_STEPS_NORM = 20
MAX_BRANCHES_NORM = 8


def task_failure_rates(trajectories: list[Trajectory]) -> dict[str, float]:
    """Per task_id: fraction of verified attempts that failed."""
    rates: dict[str, float] = {}
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_task[t.task_id].append(t)
    for task_id, group in by_task.items():
        verified = [t for t in group if t.verification is not None]
        if not verified:
            rates[task_id] = 0.5
            continue
        failed = sum(1 for t in verified if not t.verification.passed)
        rates[task_id] = failed / len(verified)
    return rates


def _branch_count(code: str) -> int:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return MAX_BRANCHES_NORM
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While,
                              ast.ExceptHandler, ast.comprehension,
                              ast.BoolOp, ast.IfExp)):
            count += 1
    return count


def difficulty(traj: Trajectory, fail_rates: dict[str, float]) -> float:
    """Difficulty in [0, 1]. Higher is harder."""
    shape = min(len(traj.steps) / MAX_STEPS_NORM, 1.0)
    task_hard = fail_rates.get(traj.task_id, 0.5)
    complexity = min(_branch_count(traj.final_code) / MAX_BRANCHES_NORM, 1.0)
    return round(0.35 * shape + 0.35 * task_hard + 0.30 * complexity, 4)


def order_curriculum(trajectories: list[Trajectory]
                     ) -> list[tuple[Trajectory, float]]:
    """Trajectories sorted easy to hard, each paired with its difficulty."""
    fail_rates = task_failure_rates(trajectories)
    scored = [(t, difficulty(t, fail_rates)) for t in trajectories]
    scored.sort(key=lambda pair: (pair[1], len(pair[0].steps)))
    return scored


def emit_curriculum_sft(trajectories: list[Trajectory],
                        include_trace: bool = False) -> list[dict]:
    """SFT rows in curriculum order, with difficulty metadata attached."""
    ordered = order_curriculum(trajectories)
    rows = []
    for order, (traj, diff) in enumerate(ordered):
        if traj.verification is None or not traj.verification.passed:
            continue
        row = to_sft_row(traj, include_trace=include_trace)
        row["difficulty"] = diff
        row["curriculum_order"] = order
        rows.append(row)
    # Re-index order over emitted rows only.
    for i, row in enumerate(rows):
        row["curriculum_order"] = i
    return rows


def curriculum_bins(rows: list[dict], n_bins: int = 4) -> list[list[dict]]:
    """Split curriculum-ordered rows into n phases for staged training."""
    if not rows:
        return [[] for _ in range(n_bins)]
    ordered = sorted(rows, key=lambda r: r.get("difficulty", 0.5))
    bins: list[list[dict]] = [[] for _ in range(n_bins)]
    for i, row in enumerate(ordered):
        bins[min(i * n_bins // len(ordered), n_bins - 1)].append(row)
    return bins


def bin_summary(bins: list[list[dict]]) -> list[dict]:
    """Mean difficulty and size per phase."""
    out = []
    for i, b in enumerate(bins):
        diffs = [r.get("difficulty", 0.5) for r in b]
        out.append({
            "phase": i + 1,
            "n": len(b),
            "mean_difficulty": round(sum(diffs) / len(diffs), 3)
            if diffs else 0.0,
        })
    return out
