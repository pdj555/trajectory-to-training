"""Quality filters for trajectories.

Filters run in order. Each returns the surviving trajectories plus a
count of how many were removed, so the dashboard can report pass rates
per filter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collector.trajectory import Trajectory
from filter.second_check import Check, relabel


@dataclass
class FilterResult:
    kept: list[Trajectory]
    removed: int
    name: str
    # For the second_check stage: trajectory id -> the reason the check gave
    # for marking that run failed. Empty for every other stage.
    detail: dict[str, str] = field(default_factory=dict)


def by_verification(trajectories: list[Trajectory],
                    keep_failures: bool = False) -> FilterResult:
    """Keep trajectories with a verification outcome.

    By default keeps only passing trajectories (for SFT). With
    keep_failures=True, keeps failing trajectories too (for preference
    pairs, where the failure becomes the rejected sample).
    """
    kept = []
    for t in trajectories:
        if t.verification is None:
            continue
        if t.verification.passed or keep_failures:
            kept.append(t)
    return FilterResult(kept=kept, removed=len(trajectories) - len(kept),
                        name="verification")


def by_length(trajectories: list[Trajectory],
              min_steps: int = 1,
              max_steps: int = 50) -> FilterResult:
    """Drop degenerate (too few steps) and rambling (too many steps) runs."""
    kept = [t for t in trajectories if min_steps <= len(t.steps) <= max_steps]
    return FilterResult(kept=kept, removed=len(trajectories) - len(kept),
                        name="length")


def dedup(trajectories: list[Trajectory]) -> FilterResult:
    """Drop exact duplicates on normalized final code.

    Keeps the first occurrence. Dedup runs before diversity caps so the
    cap counts distinct solutions, not repeated ones.
    """
    seen: set[str] = set()
    kept = []
    for t in trajectories:
        h = t.code_hash()
        if h not in seen:
            seen.add(h)
            kept.append(t)
    return FilterResult(kept=kept, removed=len(trajectories) - len(kept),
                        name="dedup")


def diversity_cap(trajectories: list[Trajectory],
                  max_per_task: int = 10) -> FilterResult:
    """Cap trajectories per task so easy tasks do not dominate the dataset.

    Within a task, prefer passing trajectories with higher test pass rates,
    then shorter trajectories (cleaner solutions first).
    """
    by_task: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_task[t.task_id].append(t)

    kept = []
    for task_id, group in by_task.items():
        group.sort(key=lambda t: (
            not (t.verification.passed if t.verification else False),
            -(t.verification.pass_rate if t.verification else 0.0),
            len(t.steps),
        ))
        kept.extend(group[:max_per_task])
    return FilterResult(kept=kept, removed=len(trajectories) - len(kept),
                        name="diversity_cap")


def run_all(trajectories: list[Trajectory],
            keep_failures: bool = False,
            min_steps: int = 1,
            max_steps: int = 50,
            max_per_task: int = 10,
            check: Check | None = None) -> tuple[list[Trajectory], list[FilterResult]]:
    """Run the full filter chain. Returns (survivors, per-filter results).

    When the caller supplies ``check``, the second check runs as the first
    stage of the chain, before the verification gate: every run that passed
    its tests is checked, and a run the check fails is marked failed, so the
    gates below treat it as a failure. The stage is reported like any other
    under the name "second_check", with ``removed`` counting the runs it
    marked failed and ``detail`` holding the reason for each one. The
    originals are not modified. With no check supplied, labels come from the
    test results alone, as before.
    """
    results = []
    current = trajectories
    if check is not None:
        report = relabel(current, check)
        current = report.trajectories
        results.append(FilterResult(kept=current, removed=len(report.flipped),
                                    name="second_check", detail=report.reasons))
    for fn in (
        lambda ts: by_verification(ts, keep_failures=keep_failures),
        lambda ts: by_length(ts, min_steps=min_steps, max_steps=max_steps),
        dedup,
        lambda ts: diversity_cap(ts, max_per_task=max_per_task),
    ):
        r = fn(current)
        results.append(r)
        current = r.kept
    return current, results
