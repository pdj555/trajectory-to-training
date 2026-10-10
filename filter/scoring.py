"""Continuous quality scoring for trajectories.

Binary filters decide keep or drop. Scoring decides how much each
trajectory is worth. Every trajectory gets a score in [0, 1] from four
components:

- verification (0.40): test pass rate, shrunk toward zero when the test
  suite is thin. A 1/1 pass is weaker evidence than a 12/12 pass.
- elegance (0.25): code complexity from the AST. Fewer branches, shallower
  nesting, and shorter code score higher. Unparseable code is penalized.
- efficiency (0.20): tool-call efficiency relative to the task-type
  median. Solving in fewer steps than typical is better. Error
  observations (retries) drag the score down.
- diversity (0.15): rare task types and unique solutions score higher.
  Exact duplicates are heavily penalized.

The score is a training weight, not a gate. Downstream, the mixer
samples proportional to score and the curriculum orders by difficulty,
not by score: hard examples can be high quality.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collector.trajectory import Trajectory

WEIGHTS = {
    "verification": 0.40,
    "elegance": 0.25,
    "efficiency": 0.20,
    "diversity": 0.15,
}

_BRANCH_NODES = (
    ast.If, ast.For, ast.AsyncFor, ast.While,
    ast.ExceptHandler, ast.With, ast.AsyncWith,
    ast.comprehension, ast.BoolOp, ast.IfExp, ast.Assert,
)


@dataclass
class QualityScore:
    trajectory_id: str
    score: float
    components: dict


def _complexity(code: str) -> dict | None:
    """Branch count, max statement-nesting depth, and code lines."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    branches = 0
    max_depth = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal branches, max_depth
        # Depth counts statement nesting only; expression nesting is noise.
        if isinstance(node, ast.stmt):
            depth += 1
            max_depth = max(max_depth, depth)
        if isinstance(node, _BRANCH_NODES):
            branches += 1
        for child in ast.iter_child_nodes(node):
            visit(child, depth)

    visit(tree, 0)
    lines = len([ln for ln in code.splitlines() if ln.strip()])
    return {"branches": branches, "depth": max_depth, "lines": lines}


def _verification_component(traj: Trajectory) -> float:
    v = traj.verification
    if v is None or v.tests_total == 0:
        return 0.0
    # Shrink toward zero when the test suite is thin: 1/1 is weak evidence.
    confidence = v.tests_total / (v.tests_total + 3.0)
    return v.pass_rate * confidence


def _elegance_component(traj: Trajectory) -> float:
    if not traj.final_code.strip():
        return 0.0
    c = _complexity(traj.final_code)
    if c is None:
        return 0.3  # unparseable: penalize, do not zero
    penalty = min(
        0.75,
        0.12 * c["branches"]
        + 0.08 * max(0, c["depth"] - 2)
        + 0.015 * max(0, c["lines"] - 25),
    )
    return 1.0 - penalty


def _efficiency_component(traj: Trajectory, median_steps: float) -> float:
    n = len(traj.steps)
    if n == 0:
        return 0.0
    base = median_steps / max(n, median_steps)  # <= median -> 1.0
    errors = sum(
        1 for s in traj.steps
        if any(m in s.observation.lower()
               for m in ("error", "traceback", "failed", "exception"))
    )
    error_rate = errors / n
    return base * (1.0 - 0.5 * error_rate)


def _diversity_component(traj: Trajectory, type_counts: dict,
                        total: int, seen_hashes: set) -> float:
    rarity = 1.0 - (type_counts.get(traj.task_type, 0) / max(total, 1))
    unique = 0.1 if traj.code_hash() in seen_hashes else 1.0
    return 0.5 * rarity + 0.5 * unique


def score_all(trajectories: list[Trajectory]) -> list[QualityScore]:
    """Score every trajectory. Batch-relative components use batch stats."""
    by_type: dict[str, list[int]] = defaultdict(list)
    for t in trajectories:
        by_type[t.task_type].append(len(t.steps))
    medians = {k: median(v) for k, v in by_type.items()}

    type_counts = defaultdict(int)
    for t in trajectories:
        type_counts[t.task_type] += 1

    seen: set[str] = set()
    scores = []
    for t in trajectories:
        components = {
            "verification": _verification_component(t),
            "elegance": _elegance_component(t),
            "efficiency": _efficiency_component(
                t, medians.get(t.task_type, 5.0)),
            "diversity": _diversity_component(
                t, type_counts, len(trajectories), seen),
        }
        seen.add(t.code_hash())
        total = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
        scores.append(QualityScore(
            trajectory_id=t.trajectory_id,
            score=round(total, 4),
            components={k: round(v, 4) for k, v in components.items()},
        ))
    return scores


def score_distribution(scores: list[QualityScore],
                       buckets: int = 10) -> list[tuple[str, int]]:
    """Histogram of scores over [0, 1]. Returns (bucket_label, count)."""
    counts = [0] * buckets
    for s in scores:
        idx = min(int(s.score * buckets), buckets - 1)
        counts[idx] += 1
    out = []
    for i, c in enumerate(counts):
        lo, hi = i / buckets, (i + 1) / buckets
        out.append((f"{lo:.1f}-{hi:.1f}", c))
    return out


def render_distribution(scores: list[QualityScore]) -> str:
    """Plain-text histogram of the score distribution."""
    dist = score_distribution(scores)
    peak = max((c for _, c in dist), default=0)
    lines = ["quality score distribution:"]
    for label, count in dist:
        bar = "#" * int(40 * count / peak) if peak else ""
        lines.append(f"  {label}  {count:4d} {bar}")
    vals = sorted(s.score for s in scores)
    if vals:
        p50 = vals[len(vals) // 2]
        lines.append(f"  median score: {p50:.3f}")
    return "\n".join(lines)


def top_k(scores: list[QualityScore], k: int) -> list[QualityScore]:
    """Highest-scoring trajectories first."""
    return sorted(scores, key=lambda s: s.score, reverse=True)[:k]
