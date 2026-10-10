"""Data-quality dashboard.

Computes the numbers that matter after a collection batch:
- verification rate overall and by task type
- trajectory length distribution (p50/p90)
- per-filter pass rates
- SFT and preference yield

Emits a plain-text report. No plotting dependencies; the numbers are
what a training-data review needs.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collector.trajectory import Trajectory
from filter.quality import FilterResult


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def verification_by_task(trajectories: list[Trajectory]) -> dict[str, dict]:
    by_type: dict[str, list[Trajectory]] = defaultdict(list)
    for t in trajectories:
        by_type[t.task_type].append(t)
    out = {}
    for task_type, group in sorted(by_type.items()):
        verified = [t for t in group if t.verification is not None]
        passed = [t for t in verified if t.verification.passed]
        out[task_type] = {
            "n": len(group),
            "verified": len(verified),
            "passed": len(passed),
            "pass_rate": len(passed) / len(verified) if verified else 0.0,
        }
    return out


def length_stats(trajectories: list[Trajectory]) -> dict:
    lengths = [len(t.steps) for t in trajectories]
    tokens = [t.prompt_tokens + t.completion_tokens for t in trajectories]
    return {
        "n": len(trajectories),
        "steps_p50": _percentile([float(x) for x in lengths], 50),
        "steps_p90": _percentile([float(x) for x in lengths], 90),
        "tokens_p50": _percentile([float(x) for x in tokens], 50),
        "tokens_p90": _percentile([float(x) for x in tokens], 90),
    }


def report(trajectories: list[Trajectory],
           filter_results: list[FilterResult],
           sft_rows: int,
           pref_pairs: int) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("TRAJECTORY DATA QUALITY REPORT")
    lines.append("=" * 60)
    lines.append(f"Raw trajectories: {len(trajectories)}")
    lines.append("")

    lines.append("Verification rate by task type:")
    for task_type, stats in verification_by_task(trajectories).items():
        lines.append(
            f"  {task_type:20s} n={stats['n']:4d} "
            f"pass_rate={stats['pass_rate']:.2f}"
        )
    lines.append("")

    ls = length_stats(trajectories)
    lines.append("Trajectory shape:")
    lines.append(f"  steps p50/p90: {ls['steps_p50']:.0f}/{ls['steps_p90']:.0f}")
    lines.append(f"  tokens p50/p90: {ls['tokens_p50']:.0f}/{ls['tokens_p90']:.0f}")
    lines.append("")

    lines.append("Filter pass rates:")
    remaining = len(trajectories)
    for r in filter_results:
        kept = len(r.kept)
        rate = kept / remaining if remaining else 0.0
        lines.append(f"  {r.name:15s} kept={kept:4d} removed={r.removed:4d} "
                     f"pass_rate={rate:.2f}")
        remaining = kept
    lines.append("")

    lines.append("Dataset yield:")
    lines.append(f"  SFT rows: {sft_rows}")
    lines.append(f"  Preference pairs: {pref_pairs}")
    lines.append("=" * 60)
    return "\n".join(lines)
