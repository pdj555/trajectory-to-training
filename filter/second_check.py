"""Second check: relabel runs that pass their tests but fail an independent check.

The pipeline's label is "did the tests pass." A run that games its tests gets that
label too, so it becomes a fine-tuning row and the "chosen" side of a preference pair.
``relabel`` runs one more check on every passing run and marks the ones that fail it
as failed. They drop out of SFT, and in preference data they move to the rejected side,
which teaches the opposite lesson.

The check is any callable ``check(traj) -> (ok, reason)``: a hidden-input comparison
against a reference, an executable spec, or a model reviewer.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collector.trajectory import Trajectory, Verification

Check = Callable[[Trajectory], tuple[bool, str]]


@dataclass
class RelabelReport:
    trajectories: list[Trajectory]
    checked: int = 0
    flipped: list[str] = field(default_factory=list)  # trajectory ids
    reasons: dict[str, str] = field(default_factory=dict)


def relabel(trajectories: list[Trajectory], check: Check) -> RelabelReport:
    """Return copies of the trajectories with test-passing runs that fail ``check`` marked failed.

    Runs that already failed their tests are left alone. The originals are not modified.
    """
    out: list[Trajectory] = []
    report = RelabelReport(trajectories=out)
    for t in trajectories:
        v = t.verification
        if v is None or not v.passed:
            out.append(t)
            continue
        report.checked += 1
        ok, reason = check(t)
        if ok:
            out.append(t)
            continue
        report.flipped.append(t.trajectory_id)
        report.reasons[t.trajectory_id] = reason
        failed = Verification(passed=False, tests_passed=v.tests_passed, tests_total=v.tests_total)
        out.append(replace(t, verification=failed))
    return report
