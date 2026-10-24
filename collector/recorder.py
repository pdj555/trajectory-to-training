"""Live trajectory recorder: emit steps as a run happens.

The Collector assembles a record by hand and rewrites the whole file on
every save. A harness that records while it runs needs the opposite shape:
an append only event log. ``Recorder`` writes one JSON line per event -
open, one per emitted step, close - appending to the file each time, so a
run is durable the moment each step lands and a crashed run still leaves
a readable record behind.

Everything written passes the scrubbing gate first: values that look like
secrets in a step's arguments or observation, or in the prompt, are
redacted before they reach the file. Trajectories feed training rows; a
secret a producing harness recorded must not ride along into them. The
gate is deliberately narrow - key prefixes, bearer tokens, and values
sitting under password-like argument names. It is not a general content
filter, and it does not touch the final code: the code is the artifact
being graded, and rewriting it would change what the label means.

Folding a recording back is ``Recorder.load``: it replays the events and
returns the same ``Trajectory`` records the Collector flow produces, so
the rest of the pipeline consumes a recording unchanged.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict

from .trajectory import Collector, ToolCall, Trajectory, Verification

REDACTED = "[REDACTED]"

# Argument names whose values are secrets by position, whatever they hold.
_SECRET_KEY_NAMES = (
    "password", "passwd", "secret", "api_key", "apikey",
    "access_token", "auth_token", "token", "credential", "private_key",
)


def scrub_text(text: str) -> str:
    """Redact secret-shaped values from a single string."""
    text = re.sub(r"sk-[A-Za-z0-9_-]{8,}", REDACTED, text)          # OpenAI / Anthropic style keys
    text = re.sub(r"AKIA[0-9A-Z]{16}", REDACTED, text)              # AWS access key ids
    text = re.sub(r"(?i)(\bBearer\s+)\S+", r"\1" + REDACTED, text)  # bearer tokens
    return text


def scrub_value(value, key: str = ""):
    """Recursively scrub a JSON-shaped value.

    A value sitting under a secret-named key is replaced outright; every
    string anywhere else passes through ``scrub_text``.
    """
    if any(name in key.lower() for name in _SECRET_KEY_NAMES):
        return REDACTED
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, dict):
        return {k: scrub_value(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_value(v) for v in value]
    return value


def scrub_step(step: ToolCall) -> ToolCall:
    """Return a copy of ``step`` with its arguments and observation scrubbed."""
    return ToolCall(
        tool=step.tool,
        arguments=scrub_value(step.arguments),
        observation=scrub_text(step.observation),
        latency_ms=step.latency_ms,
    )


class Recorder:
    """Append only trajectory recorder bound to one JSONL file.

    Usage:
        recorder = Recorder("trajectories.jsonl")
        traj = recorder.open(task_id="t1", task_type="bugfix", prompt="...")
        recorder.emit(traj, tool="edit", arguments={...}, observation="...")
        recorder.close(traj, final_code="...", verification=Verification(True, 5, 5))
        trajectories = Recorder.load("trajectories.jsonl")
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._open: dict[str, Trajectory] = {}
        self.trajectories: list[Trajectory] = []  # closed, in close order

    def open(self, task_id: str, task_type: str, prompt: str,
             model: str = "") -> Trajectory:
        traj = Trajectory(
            trajectory_id=uuid.uuid4().hex[:12],
            task_id=task_id,
            task_type=task_type,
            prompt=scrub_text(prompt),
            model=model,
        )
        self._open[traj.trajectory_id] = traj
        self._append({
            "event": "open",
            "trajectory_id": traj.trajectory_id,
            "task_id": traj.task_id,
            "task_type": traj.task_type,
            "prompt": traj.prompt,
            "model": traj.model,
            "created_at": traj.created_at,
        })
        return traj

    def emit(self, traj: Trajectory, step: ToolCall | None = None, *,
             tool: str | None = None, arguments: dict | None = None,
             observation: str = "", latency_ms: float = 0.0) -> ToolCall:
        """Record one step on an open trajectory, as it happens.

        Accepts either a ready ``ToolCall`` or the step's fields. The step
        is scrubbed before it is stored or written; the returned step is
        the scrubbed one that landed in the record.
        """
        if step is None:
            if tool is None:
                raise ValueError("emit needs a step or a tool name")
            step = ToolCall(tool=tool, arguments=arguments or {},
                            observation=observation, latency_ms=latency_ms)
        step = scrub_step(step)
        traj.steps.append(step)
        self._append({
            "event": "step",
            "trajectory_id": traj.trajectory_id,
            "step": asdict(step),
        })
        return step

    def close(self, traj: Trajectory, final_code: str,
              verification: Verification,
              prompt_tokens: int = 0,
              completion_tokens: int = 0,
              total_latency_ms: float = 0.0) -> Trajectory:
        """Finish a trajectory and append its closing record.

        The closing line carries the full record in the same shape
        ``Collector.save_jsonl`` writes, so a folded recording feeds the
        pipeline exactly like a hand assembled collection.
        """
        traj.final_code = final_code
        traj.verification = verification
        traj.prompt_tokens = prompt_tokens
        traj.completion_tokens = completion_tokens
        traj.total_latency_ms = total_latency_ms
        self._open.pop(traj.trajectory_id, None)
        self.trajectories.append(traj)
        self._append({
            "event": "close",
            "trajectory_id": traj.trajectory_id,
            "trajectory": traj.to_dict(),
        })
        return traj

    @classmethod
    def load(cls, path: str) -> list[Trajectory]:
        """Fold a recording back into trajectories, in open order.

        A closed trajectory folds to its closing record. One that never
        closed - the harness died mid run - folds from its open event and
        the steps emitted before the end of the file, with no verification
        record.
        """
        order: list[str] = []
        folded: dict[str, Trajectory] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                kind = event.get("event")
                traj_id = event.get("trajectory_id")
                if kind == "open":
                    folded[traj_id] = Trajectory(
                        trajectory_id=traj_id,
                        task_id=event["task_id"],
                        task_type=event.get("task_type", "unknown"),
                        prompt=event["prompt"],
                        model=event.get("model", ""),
                        created_at=event.get("created_at", time.time()),
                    )
                    order.append(traj_id)
                elif kind == "step" and traj_id in folded:
                    folded[traj_id].steps.append(ToolCall(**event["step"]))
                elif kind == "close":
                    folded[traj_id] = Trajectory.from_dict(event["trajectory"])
                    if traj_id not in order:
                        order.append(traj_id)
        return [folded[traj_id] for traj_id in order]

    @classmethod
    def collector(cls, path: str) -> Collector:
        """Fold a recording into a Collector, ready for the pipeline."""
        collector = Collector()
        collector.trajectories = cls.load(path)
        return collector

    def _append(self, event: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def __len__(self) -> int:
        return len(self.trajectories)
