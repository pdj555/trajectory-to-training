"""Trajectory repair: turn failures into training data.

Failed trajectories are usually discarded. That wastes most of the
collection budget, because failures outnumber passes. Repair recovers
two kinds of value:

1. Sibling repair. When a task has at least one passing trajectory, a
   failing trajectory on the same task is repaired by keeping its prefix
   up to the failure point and continuing with the passing sibling's
   suffix. The repaired trajectory is marked and carries a repair note,
   so downstream consumers can weight it differently.

2. Near-miss rescue. Failures with a high partial pass rate (default
   >= 0.6) are rescued as hard negatives: they are almost right, which
   makes them the most informative rejected samples for preference
   training. The pass rate travels with them as a soft label.

Repair never invents code. The corrected continuation always comes from
a verified passing trajectory on the same task.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collector.trajectory import Trajectory

ERROR_MARKERS = ("error", "traceback", "failed", "exception", "timeout")

NEAR_MISS_THRESHOLD = 0.6


@dataclass
class RepairReport:
    repaired: list[Trajectory]
    near_misses: list[Trajectory]
    unrecoverable: int


def failure_point(traj: Trajectory) -> int | None:
    """Index of the first step whose observation signals failure.

    Returns None for semantic failures: the trajectory ran clean but the
    final code failed its tests. Returns len(steps) when no code was
    produced at all.
    """
    for i, step in enumerate(traj.steps):
        obs = step.observation.lower()
        if any(m in obs for m in ERROR_MARKERS):
            return i
    if not traj.final_code.strip():
        return len(traj.steps)
    return None


def repair_from_sibling(failed: Trajectory,
                        sibling: Trajectory) -> Trajectory:
    """Build a repaired trajectory from a failing one and a passing sibling.

    Keeps the failed trajectory's prefix up to its failure point, then
    continues with the sibling's full step sequence and verified final
    code. The result is marked repaired and cites its source.
    """
    fp = failure_point(failed)
    prefix = failed.steps[:fp] if fp is not None else list(failed.steps)
    return Trajectory(
        trajectory_id=failed.trajectory_id + "-repaired",
        task_id=failed.task_id,
        task_type=failed.task_type,
        prompt=failed.prompt,
        steps=prefix + list(sibling.steps),
        final_code=sibling.final_code,
        verification=sibling.verification,
        model=failed.model,
        prompt_tokens=failed.prompt_tokens + sibling.prompt_tokens,
        completion_tokens=sibling.completion_tokens,
        total_latency_ms=failed.total_latency_ms + sibling.total_latency_ms,
        repaired=True,
        repair_note=(
            f"sibling-repair from {sibling.trajectory_id}; "
            f"failure at step {fp}; kept {len(prefix)} prefix steps"
        ),
    )


def is_near_miss(traj: Trajectory,
                 threshold: float = NEAR_MISS_THRESHOLD) -> bool:
    """A failing trajectory that is almost right: high partial pass rate."""
    v = traj.verification
    return (v is not None and not v.passed
            and v.tests_total > 0 and v.pass_rate >= threshold)


def repair_batch(trajectories: list[Trajectory],
                 near_miss_threshold: float = NEAR_MISS_THRESHOLD
                 ) -> RepairReport:
    """Repair every recoverable failure in a batch.

    Groups by task. For each failing trajectory: if a passing sibling
    exists on the same task, repair it; else if it is a near miss,
    rescue it as a hard negative; else count it unrecoverable.
    """
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_task[t.task_id].append(t)

    repaired: list[Trajectory] = []
    near_misses: list[Trajectory] = []
    unrecoverable = 0

    for task_id, group in by_task.items():
        passes = [t for t in group
                  if t.verification is not None and t.verification.passed]
        fails = [t for t in group
                 if t.verification is not None and not t.verification.passed]
        if not passes:
            for f in fails:
                if f.verification.pass_rate >= near_miss_threshold \
                        and f.verification.tests_total > 0:
                    near_misses.append(f)
                else:
                    unrecoverable += 1
            continue
        # Best sibling: highest pass rate, then fewest steps.
        passes.sort(key=lambda t: (-t.verification.pass_rate, len(t.steps)))
        best = passes[0]
        for f in fails:
            if f is best:
                continue
            repaired.append(repair_from_sibling(f, best))

    return RepairReport(repaired=repaired, near_misses=near_misses,
                        unrecoverable=unrecoverable)
