"""The live recorder: steps land as they happen, appends never rewrite, secrets never reach the file."""

from __future__ import annotations

import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from collector import Recorder, ToolCall, Verification  # noqa: E402

FAKE_ANTHROPIC_KEY = "sk-ant-api03-FAKEFAKEFAKEFAKEFAKEFAKE"
FAKE_AWS_KEY = "AKIAFAKEFAKEFAKEFAKE"  # AKIA plus 16 characters
FAKE_BEARER_TOKEN = "FAKEtokenFAKEtoken123"


def test_emit_appends_without_rewriting(tmp_path):
    path = str(tmp_path / "log.jsonl")
    recorder = Recorder(path)
    traj = recorder.open("t1", "bugfix", "fix it")
    recorder.emit(traj, tool="read", arguments={"path": "a.py"}, observation="ok")
    with open(path, "rb") as f:
        prefix = f.read()
    recorder.emit(traj, tool="edit", arguments={"path": "a.py"}, observation="done")
    recorder.close(traj, "code", Verification(True, 1, 1))
    with open(path, "rb") as f:
        full = f.read()
    assert full.startswith(prefix) and len(full) > len(prefix)
    assert len(full.decode().splitlines()) == 4  # open, two steps, close
    # a second recorded run appends after the first, leaving it untouched
    traj2 = recorder.open("t2", "feature", "build it")
    recorder.close(traj2, "code2", Verification(False, 0, 1))
    with open(path, "rb") as f:
        assert f.read().startswith(full)


def test_emit_round_trips_into_the_collector_shape(tmp_path):
    path = str(tmp_path / "log.jsonl")
    recorder = Recorder(path)
    traj = recorder.open("t1", "bugfix", "fix the parser", model="m")
    recorder.emit(traj, ToolCall(tool="read", arguments={"path": "p.py"},
                                 observation="saw it", latency_ms=3.0))
    recorder.emit(traj, tool="edit", arguments={"path": "p.py"},
                  observation="patched", latency_ms=5.0)
    recorder.close(traj, "def f(): ...", Verification(True, 4, 5),
                   prompt_tokens=10, completion_tokens=20, total_latency_ms=8.0)

    # the same run assembled by hand through the Collector flow
    from collector import Collector
    hand = Collector()
    want = hand.open("t1", "bugfix", "fix the parser", model="m")
    want.steps.append(ToolCall(tool="read", arguments={"path": "p.py"},
                               observation="saw it", latency_ms=3.0))
    want.steps.append(ToolCall(tool="edit", arguments={"path": "p.py"},
                               observation="patched", latency_ms=5.0))
    hand.close(want, "def f(): ...", Verification(True, 4, 5),
               prompt_tokens=10, completion_tokens=20, total_latency_ms=8.0)

    loaded = Recorder.load(path)
    assert len(loaded) == 1
    got, expected = loaded[0].to_dict(), want.to_dict()
    for name in ("task_id", "task_type", "prompt", "steps", "final_code",
                 "verification", "model", "prompt_tokens",
                 "completion_tokens", "total_latency_ms"):
        assert got[name] == expected[name]
    # and the folded records drop straight into a Collector for the pipeline
    assert len(Recorder.collector(path)) == 1


def test_planted_secrets_never_reach_the_file(tmp_path):
    path = str(tmp_path / "log.jsonl")
    recorder = Recorder(path)
    traj = recorder.open("t1", "bugfix", f"debug with {FAKE_ANTHROPIC_KEY}")
    recorder.emit(
        traj,
        tool="http",
        arguments={"aws_key": FAKE_AWS_KEY, "password": "hunter2",
                   "url": "https://example.com"},
        observation=f"auth failed for Bearer {FAKE_BEARER_TOKEN}; "
                    f"key {FAKE_ANTHROPIC_KEY} rejected",
    )
    recorder.close(traj, "code", Verification(True, 1, 1))
    with open(path) as f:
        raw = f.read()
    for secret in (FAKE_ANTHROPIC_KEY, FAKE_AWS_KEY, "hunter2", FAKE_BEARER_TOKEN):
        assert secret not in raw
    assert "[REDACTED]" in raw
    loaded = Recorder.load(path)
    step = loaded[0].steps[0]
    assert step.arguments["password"] == "[REDACTED]"
    assert step.arguments["aws_key"] == "[REDACTED]"
    assert step.arguments["url"] == "https://example.com"  # non-secret data survives


def test_unclosed_run_folds_from_its_events(tmp_path):
    path = str(tmp_path / "log.jsonl")
    recorder = Recorder(path)
    traj = recorder.open("t1", "bugfix", "fix it")
    recorder.emit(traj, tool="read", arguments={}, observation="halfway")
    # no close: the harness died here
    loaded = Recorder.load(path)
    assert len(loaded) == 1
    assert loaded[0].verification is None and len(loaded[0].steps) == 1
