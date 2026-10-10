"""Worked example: full pipeline on synthetic trajectories.

Synthesizes 200 trajectories across 4 task types with realistic failure
modes, runs collect -> filter -> emit -> report, and prints the quality
dashboard.

Expected output (seeded, deterministic):
- 200 raw trajectories
- verification filter keeps 136 (68% pass rate by design)
- length filter keeps all 136 (all within bounds by design)
- dedup removes 13 (duplicate solutions by design)
- diversity cap keeps 123 (no task over the cap in this seed)
- SFT rows: 123, preference pairs: 37

Run: python3 examples/worked_example.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collector import Collector, ToolCall, Verification
from filter import quality as qf
from datasets import sft as sft_mod
from datasets import preference as pref_mod
from datasets import split as split_mod
from dashboard import metrics as dash

rng = random.Random(7)

TASK_TYPES = ["bugfix", "feature", "refactor", "testgen"]

PROMPTS = {
    "bugfix": "Fix the off-by-one error in paginate() so the last page is not dropped.",
    "feature": "Add a retry decorator with exponential backoff to the http client.",
    "refactor": "Extract the validation logic from submit() into a standalone function.",
    "testgen": "Write unit tests for the LRU cache covering eviction and TTL expiry.",
}

FAILURE_MODES = [
    "empty_final_code",
    "wrong_function_signature",
    "tests_partially_passing",
    "infinite_tool_loop",
]


def synth_trajectory(collector: Collector, i: int) -> None:
    task_type = TASK_TYPES[i % len(TASK_TYPES)]
    # 12 distinct tasks per type, ~4 attempts each, so preference pairs form.
    task_id = f"{task_type}-{(i // len(TASK_TYPES)) % 12}"
    traj = collector.open(
        task_id=task_id,
        task_type=task_type,
        prompt=PROMPTS[task_type],
        model="synth-agent-1",
    )

    n_steps = rng.randint(2, 12)
    for s in range(n_steps):
        traj.steps.append(ToolCall(
            tool=rng.choice(["read", "edit", "run_tests", "search"]),
            arguments={"target": f"module_{s}.py"},
            observation="ok" if rng.random() > 0.2 else "error: retry",
            latency_ms=rng.uniform(50, 2000),
        ))

    roll = rng.random()
    if roll < 0.70:
        passed, total = 5, 5
        mode = "clean_pass"
    elif roll < 0.80:
        passed, total = rng.randint(2, 4), 5
        mode = "tests_partially_passing"
    elif roll < 0.87:
        passed, total = 0, 5
        mode = "wrong_function_signature"
    elif roll < 0.93:
        passed, total = 0, 0
        mode = "empty_final_code"
    else:
        passed, total = 0, 5
        mode = "infinite_tool_loop"

    # 7% of passes are exact duplicates of an earlier solution.
    code = f"def solution_{task_id}():\n    return {i % 13}\n"
    if mode == "clean_pass" and rng.random() < 0.10:
        code = "def solution_bugfix-0():\n    return 0\n"

    collector.close(
        traj,
        final_code="" if mode == "empty_final_code" else code,
        verification=Verification(
            passed=(passed == total and total > 0),
            tests_passed=passed,
            tests_total=total,
        ),
        prompt_tokens=rng.randint(200, 1500),
        completion_tokens=rng.randint(100, 3000),
        total_latency_ms=sum(s.latency_ms for s in traj.steps),
    )


def main() -> None:
    collector = Collector()
    for i in range(200):
        synth_trajectory(collector, i)
    raw = collector.trajectories
    print(f"Collected {len(raw)} trajectories\n")

    # SFT path: passes only.
    sft_kept, sft_filters = qf.run_all(raw, keep_failures=False)
    sft_rows = sft_mod.emit_sft(sft_kept)

    # Preference path: passes and failures.
    pref_kept, _ = qf.run_all(raw, keep_failures=True)
    pairs = pref_mod.build_pairs(pref_kept)

    train, val = split_mod.stratified_split(sft_rows)

    print(dash.report(raw, sft_filters, len(sft_rows), len(pairs)))
    print(f"\nTrain/val split: {len(train)} train, {len(val)} val")

    # Persist the example outputs next to this script.
    outdir = os.path.dirname(os.path.abspath(__file__))
    sft_mod.save_jsonl(sft_rows, os.path.join(outdir, "sft_example.jsonl"))
    pref_mod.save_jsonl(pairs, os.path.join(outdir, "pref_example.jsonl"))
    print(f"Wrote sft_example.jsonl ({len(sft_rows)} rows) and "
          f"pref_example.jsonl ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
