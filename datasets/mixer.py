"""Data mixing: what ratio of each task type goes into training.

A dataset with four task types is not automatically a good blend. If
90% of rows are easy bugfixes, the model learns bugfixes and little
else. The mixer computes a sampling plan over task types and then
samples the best rows per type.

Strategies:

- uniform: equal rows per task type. The neutral default.
- weakness: weight inversely to the task type's pass rate. Types the
  agent is bad at get more rows, because that is where the headroom is.
- target: an explicit capability profile, e.g. {"bugfix": 0.2,
  "feature": 0.4, ...}. Fractions are normalized.

Within a type, rows are taken in quality-score order when scores are
supplied, otherwise by pass rate then brevity. Sampling never invents
rows: a type with fewer trajectories than its allocation contributes
what it has, and the remainder is redistributed.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collector.trajectory import Trajectory
from filter.scoring import QualityScore

STRATEGIES = ("uniform", "weakness", "target")


@dataclass
class MixPlan:
    strategy: str
    counts: dict  # task_type -> rows allocated
    weights: dict  # task_type -> target fraction


def _pass_rate_by_type(trajectories: list[Trajectory]) -> dict[str, float]:
    by_type: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_type[t.task_type].append(t)
    out = {}
    for task_type, group in by_type.items():
        verified = [t for t in group if t.verification is not None]
        passed = [t for t in verified if t.verification.passed]
        out[task_type] = len(passed) / len(verified) if verified else 0.0
    return out


def plan_mix(trajectories: list[Trajectory],
             total: int | None = None,
             strategy: str = "weakness",
             target: dict[str, float] | None = None) -> MixPlan:
    """Compute per-task-type row allocations."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    by_type: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_type[t.task_type].append(t)
    types = sorted(by_type)
    if total is None:
        total = len(trajectories)

    if strategy == "uniform":
        weights = {k: 1.0 / len(types) for k in types}
    elif strategy == "weakness":
        rates = _pass_rate_by_type(trajectories)
        raw = {k: (1.0 - rates.get(k, 0.0)) + 0.1 for k in types}
        s = sum(raw.values())
        weights = {k: v / s for k, v in raw.items()}
    else:  # target
        if not target:
            raise ValueError("target strategy needs a target profile")
        s = sum(target.values())
        weights = {k: target.get(k, 0.0) / s for k in types}

    counts = {k: int(round(weights[k] * total)) for k in types}
    # Fix rounding drift on the largest type.
    drift = total - sum(counts.values())
    if drift:
        biggest = max(types, key=lambda k: weights[k])
        counts[biggest] += drift
    # Cap at availability; redistribute the remainder.
    for k in types:
        counts[k] = min(counts[k], len(by_type[k]))
    shortfall = total - sum(counts.values())
    if shortfall > 0:
        # Give remainder to types with headroom, in weight order.
        for k in sorted(types, key=lambda k: -weights[k]):
            room = len(by_type[k]) - counts[k]
            take = min(room, shortfall)
            counts[k] += take
            shortfall -= take
            if shortfall <= 0:
                break
    return MixPlan(strategy=strategy, counts=counts, weights=weights)


def apply_mix(trajectories: list[Trajectory],
              plan: MixPlan,
              scores: list[QualityScore] | None = None,
              seed: int = 42) -> list[Trajectory]:
    """Sample trajectories per the mix plan.

    Within each type, prefer higher quality scores when available;
    otherwise prefer passing trajectories, then shorter ones.
    """
    rng = random.Random(seed)
    score_by_id = {s.trajectory_id: s.score for s in scores} if scores else {}
    by_type: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_type[t.task_type].append(t)

    def rank_key(t: Trajectory):
        if t.trajectory_id in score_by_id:
            return (-score_by_id[t.trajectory_id], len(t.steps))
        passed = t.verification.passed if t.verification else False
        rate = t.verification.pass_rate if t.verification else 0.0
        return (not passed, -rate, len(t.steps))

    mixed: list[Trajectory] = []
    for task_type in sorted(plan.counts):
        group = sorted(by_type.get(task_type, []), key=rank_key)
        mixed.extend(group[:plan.counts[task_type]])
    rng.shuffle(mixed)
    return mixed


def render_plan(plan: MixPlan) -> str:
    lines = [f"mix plan (strategy={plan.strategy}):"]
    for k in sorted(plan.counts):
        lines.append(f"  {k:12s} weight={plan.weights[k]:.2f} "
                     f"rows={plan.counts[k]}")
    lines.append(f"  total rows: {sum(plan.counts.values())}")
    return "\n".join(lines)
