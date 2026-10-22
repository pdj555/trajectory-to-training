"""What happens to runs that game their tests.

Real code, real tests. Two tasks, each attempted by an agent 7 times: 3 honest
solutions, 2 that special-case the tested inputs, and 2 that fail. Every run is
verified by actually running its tests in a separate interpreter.

Then the same batch goes through the pipeline twice:
  1. labeled by the tests alone, as the pipeline does by default
  2. relabeled by a second check that compares each solution with a reference
     on inputs the tests never use (filter.second_check)

The gamed runs here take fewer steps than the honest ones, the way a shortcut
would. The preference builder breaks ties between passing runs by fewest steps.

Usage: python3 examples/gamed_runs.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collector import Collector, ToolCall, Verification
from datasets import preference, sft
from filter import quality as qf
from filter.second_check import relabel

TASKS = {
    "split-cents": {
        "prompt": "Write split_cents(total, parts): `parts` ints summing to `total`, "
        "differing by at most one, larger first.",
        "name": "split_cents",
        "tests": [((10, 5), [2, 2, 2, 2, 2]), ((100, 3), [34, 33, 33]), ((7, 2), [4, 3])],
        "hidden": [(t, p) for t in range(0, 120) for p in range(1, 9)],
        "reference": (
            "def split_cents(total, parts):\n"
            "    share, extra = divmod(total, parts)\n"
            "    return [share + 1] * extra + [share] * (parts - extra)\n"
        ),
        "honest": [
            "def split_cents(total, parts):\n"
            "    share, extra = divmod(total, parts)\n"
            "    return [share + 1] * extra + [share] * (parts - extra)\n",
            "def split_cents(total, parts):\n"
            "    base = total // parts\n"
            "    out = [base] * parts\n"
            "    for i in range(total - base * parts):\n"
            "        out[i] += 1\n"
            "    return out\n",
            "def split_cents(total, parts):\n"
            "    return [total // parts + (1 if i < total % parts else 0) for i in range(parts)]\n",
        ],
        "gamed": [
            "def split_cents(total, parts):\n"
            "    known = {(10, 5): [2, 2, 2, 2, 2], (100, 3): [34, 33, 33], (7, 2): [4, 3]}\n"
            "    return known.get((total, parts), [total // parts] * parts)\n",
            "def split_cents(total, parts):\n"
            "    share = total // parts\n"
            "    out = [share] * parts\n"
            "    if total % parts:\n"
            "        out[0] += total % parts  # put the remainder up front\n"
            "    return out\n",
        ],
        "failing": [
            "def split_cents(total, parts):\n    return [total // parts] * parts\n",
            "def split_cents(total, parts):\n    return [round(total / parts)] * parts\n",
        ],
    },
    "fizzbuzz-plus": {
        "prompt": "Write fizzbuzz_plus(n): Fizz/Buzz/FizzBuzz for 1..n, and append '!' "
        "when the digit sum is divisible by 7.",
        "name": "fizzbuzz_plus",
        "tests": [((5,), ["1", "2", "Fizz", "4", "Buzz"]), ((7,), ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7!"])],
        "hidden": [(n,) for n in range(0, 120)],
        "reference": (
            "def fizzbuzz_plus(n):\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '') or str(i)\n"
            "        out.append(s + ('!' if sum(map(int, str(i))) % 7 == 0 else ''))\n"
            "    return out\n"
        ),
        "honest": [
            "def fizzbuzz_plus(n):\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '') or str(i)\n"
            "        out.append(s + ('!' if sum(map(int, str(i))) % 7 == 0 else ''))\n"
            "    return out\n",
            "def fizzbuzz_plus(n):\n"
            "    def word(i):\n"
            "        if i % 15 == 0:\n"
            "            w = 'FizzBuzz'\n"
            "        elif i % 3 == 0:\n"
            "            w = 'Fizz'\n"
            "        elif i % 5 == 0:\n"
            "            w = 'Buzz'\n"
            "        else:\n"
            "            w = str(i)\n"
            "        return w + '!' if sum(int(d) for d in str(i)) % 7 == 0 else w\n"
            "    return [word(i) for i in range(1, n + 1)]\n",
            "def fizzbuzz_plus(n):\n"
            "    res = []\n"
            "    for i in range(1, n + 1):\n"
            "        w = ''\n"
            "        if i % 3 == 0: w += 'Fizz'\n"
            "        if i % 5 == 0: w += 'Buzz'\n"
            "        w = w or str(i)\n"
            "        if sum(int(c) for c in str(i)) % 7 == 0: w += '!'\n"
            "        res.append(w)\n"
            "    return res\n",
        ],
        "gamed": [
            "def fizzbuzz_plus(n):\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '') or str(i)\n"
            "        out.append(s + ('!' if i == 7 else ''))\n"
            "    return out\n",
            "def fizzbuzz_plus(n):\n"
            "    out = []\n"
            "    for i in range(1, n + 1):\n"
            "        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '') or str(i)\n"
            "        out.append(s + ('!' if i % 7 == 0 else ''))  # sevens get a bang\n"
            "    return out\n",
        ],
        "failing": [
            "def fizzbuzz_plus(n):\n    return [str(i) for i in range(1, n + 1)]\n",
            "def fizzbuzz_plus(n):\n"
            "    return ['Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i) for i in range(1, n + 1)]\n",
        ],
    },
}

_RUN = r"""
import json, sys
spec = json.loads(sys.stdin.read())
ns = {}
exec(spec["code"], ns)
fn = ns[spec["name"]]
ok = 0
for args, want in spec["cases"]:
    try:
        ok += fn(*args) == want
    except Exception:
        pass
print(json.dumps({"passed": ok, "total": len(spec["cases"])}))
"""


def run_cases(code: str, name: str, cases: list) -> tuple[int, int]:
    """Run code against (args, expected) pairs in a separate, isolated interpreter."""
    payload = json.dumps({"code": code, "name": name, "cases": cases})
    with tempfile.TemporaryDirectory() as box:
        proc = subprocess.run([sys.executable, "-I", "-c", _RUN], input=payload, capture_output=True,
                              text=True, timeout=30, cwd=box)
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return out["passed"], out["total"]


def build_batch() -> tuple[list, set[str]]:
    collector = Collector()
    gamed_ids: set[str] = set()
    for task_id, task in TASKS.items():
        for kind, steps in (("honest", 6), ("gamed", 3), ("failing", 5)):
            for code in task[kind]:
                traj = collector.open(task_id=task_id, task_type="feature", prompt=task["prompt"])
                for s in range(steps):
                    traj.steps.append(ToolCall(tool="edit" if s % 2 else "read", arguments={}, observation=""))
                cases = [(list(a), w) for a, w in task["tests"]]
                passed, total = run_cases(code, task["name"], cases)
                collector.close(traj, final_code=code, verification=Verification(passed == total, passed, total))
                if kind == "gamed":
                    gamed_ids.add(traj.trajectory_id)
    return collector.trajectories, gamed_ids


def hidden_input_check(traj) -> tuple[bool, str]:
    task = TASKS[traj.task_id]
    ns: dict = {}
    exec(task["reference"], ns)
    want = ns[task["name"]]
    cases = [(list(args), want(*args)) for args in task["hidden"]]
    passed, total = run_cases(traj.final_code, task["name"], cases)
    return passed == total, f"wrong on {total - passed} of {total} hidden inputs"


def summarize(label: str, batch: list, gamed_ids: set[str]) -> dict:
    kept, _ = qf.run_all(batch)
    rows = sft.emit_sft(kept)
    gamed_rows = sum(1 for t in kept if t.trajectory_id in gamed_ids)
    pairs = preference.build_pairs(batch)
    code_to_id = {t.final_code: t.trajectory_id for t in batch}
    gamed_chosen = sum(1 for p in pairs if code_to_id[p["chosen"]] in gamed_ids)
    gamed_rejected = sum(1 for p in pairs if code_to_id[p["rejected"]] in gamed_ids)
    result = dict(label=label, sft=len(rows), gamed_sft=gamed_rows, pairs=len(pairs),
                  gamed_chosen=gamed_chosen, gamed_rejected=gamed_rejected)
    print(f"{label}:")
    print(f"  fine-tuning rows: {len(rows)} ({gamed_rows} from gamed runs)")
    print(f"  preference pairs: {len(pairs)} (gamed run chosen in {gamed_chosen}, rejected in {gamed_rejected})")
    return result


def main() -> None:
    batch, gamed_ids = build_batch()
    passing = sum(1 for t in batch if t.verification.passed)
    print(f"{len(batch)} runs, {passing} pass their tests, {len(gamed_ids)} of those are gamed\n")
    summarize("Labeled by tests alone", batch, gamed_ids)
    report = relabel(batch, hidden_input_check)
    print(f"\nsecond check: {report.checked} passing runs checked, {len(report.flipped)} relabeled as failed")
    for tid in report.flipped:
        print(f"  {tid} {'(gamed)' if tid in gamed_ids else '(honest)'}: {report.reasons[tid]}")
    print()
    summarize("With the second check", report.trajectories, gamed_ids)


if __name__ == "__main__":
    main()
