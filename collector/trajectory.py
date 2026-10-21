"""Trajectory schema and collection API.

A trajectory is one attempt at one task: the prompt, the sequence of
tool calls and observations, the final code, and the verification outcome.
The verification outcome is the label. Everything downstream keys off it.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field, asdict


@dataclass
class ToolCall:
    """A single tool invocation inside a trajectory."""
    tool: str
    arguments: dict
    observation: str
    latency_ms: float = 0.0


@dataclass
class Verification:
    """Ground-truth outcome for the trajectory's final code."""
    passed: bool
    tests_passed: int
    tests_total: int

    @property
    def pass_rate(self) -> float:
        if self.tests_total == 0:
            return 0.0
        return self.tests_passed / self.tests_total


@dataclass
class Trajectory:
    """One agent attempt at one task, with its verification outcome."""
    trajectory_id: str
    task_id: str
    task_type: str
    prompt: str
    steps: list[ToolCall] = field(default_factory=list)
    final_code: str = ""
    verification: Verification | None = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: float = 0.0
    created_at: float = field(default_factory=time.time)
    # Set by filter.repair: a failed run rebuilt from a verified sibling,
    # and a note naming the source so the provenance stays auditable.
    repaired: bool = False
    repair_note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        steps = [ToolCall(**s) for s in d.get("steps", [])]
        v = d.get("verification")
        verification = Verification(**v) if v else None
        return cls(
            trajectory_id=d["trajectory_id"],
            task_id=d["task_id"],
            task_type=d.get("task_type", "unknown"),
            prompt=d["prompt"],
            steps=steps,
            final_code=d.get("final_code", ""),
            verification=verification,
            model=d.get("model", ""),
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_latency_ms=d.get("total_latency_ms", 0.0),
            created_at=d.get("created_at", time.time()),
            repaired=d.get("repaired", False),
            repair_note=d.get("repair_note", ""),
        )

    def code_hash(self) -> str:
        """Stable hash of normalized final code, for dedup."""
        normalized = "\n".join(
            line.rstrip() for line in self.final_code.strip().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class Collector:
    """In-memory trajectory collector with JSONL persistence.

    Usage:
        collector = Collector()
        traj = collector.open(task_id="t1", task_type="bugfix", prompt="...")
        traj.steps.append(ToolCall(tool="edit", arguments={...}, observation="..."))
        collector.close(traj, final_code="...", verification=Verification(True, 5, 5))
        collector.save_jsonl("trajectories.jsonl")
    """

    def __init__(self) -> None:
        self.trajectories: list[Trajectory] = []

    def open(self, task_id: str, task_type: str, prompt: str,
             model: str = "") -> Trajectory:
        return Trajectory(
            trajectory_id=uuid.uuid4().hex[:12],
            task_id=task_id,
            task_type=task_type,
            prompt=prompt,
            model=model,
        )

    def close(self, traj: Trajectory, final_code: str,
              verification: Verification,
              prompt_tokens: int = 0,
              completion_tokens: int = 0,
              total_latency_ms: float = 0.0) -> Trajectory:
        traj.final_code = final_code
        traj.verification = verification
        traj.prompt_tokens = prompt_tokens
        traj.completion_tokens = completion_tokens
        traj.total_latency_ms = total_latency_ms
        self.trajectories.append(traj)
        return traj

    def save_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            for traj in self.trajectories:
                f.write(json.dumps(traj.to_dict()) + "\n")

    @classmethod
    def load_jsonl(cls, path: str) -> "Collector":
        collector = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    collector.trajectories.append(Trajectory.from_dict(json.loads(line)))
        return collector

    def __len__(self) -> int:
        return len(self.trajectories)
