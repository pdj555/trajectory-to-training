"""SFT dataset emission.

Converts passing trajectories to chat-format SFT rows. Two modes:
- final_code: the assistant message is just the final code. Teaches the product.
- trace: the assistant message includes the tool-call trace. Teaches the process.

Rows are JSONL with {"messages": [{"role": "user", ...}, {"role": "assistant", ...}],
"task_id", "task_type"}.
"""

from __future__ import annotations

import json

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from collector.trajectory import Trajectory


def _format_trace(traj: Trajectory) -> str:
    lines = []
    for i, step in enumerate(traj.steps, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in step.arguments.items())
        lines.append(f"[step {i}] {step.tool}({args})")
        obs = step.observation.strip().replace("\n", " ")
        if len(obs) > 300:
            obs = obs[:300] + "..."
        lines.append(f"  -> {obs}")
    lines.append("")
    lines.append("Final code:")
    lines.append(traj.final_code)
    return "\n".join(lines)


def to_sft_row(traj: Trajectory, include_trace: bool = False) -> dict:
    content = _format_trace(traj) if include_trace else traj.final_code
    return {
        "messages": [
            {"role": "user", "content": traj.prompt},
            {"role": "assistant", "content": content},
        ],
        "task_id": traj.task_id,
        "task_type": traj.task_type,
    }


def emit_sft(trajectories: list[Trajectory],
             include_trace: bool = False) -> list[dict]:
    """Emit SFT rows. Callers should pass verification-filtered trajectories."""
    rows = []
    for t in trajectories:
        if t.verification is None or not t.verification.passed:
            continue
        rows.append(to_sft_row(t, include_trace=include_trace))
    return rows


def save_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
