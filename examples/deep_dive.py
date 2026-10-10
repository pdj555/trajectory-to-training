"""Deep dive: the six depth features on synthetic trajectories.

Demonstrates, end to end on one seeded batch of 200 trajectories:

1. Quality scoring: continuous scores replace keep/drop. Prints the
   score distribution and the top 3 trajectories with components.
2. Near-dedup: structural AST dedup catches renames that exact hashing
   misses. Prints how many extra duplicates it finds.
3. Repair: failed trajectories with a passing sibling get repaired;
   near misses are rescued. Prints the recovery counts.
4. Curriculum: SFT rows ordered easy to hard. Prints the easiest and
   hardest rows plus per-phase difficulty.
5. Mixing: weakness-weighted blend across task types. Prints the plan.
6. Flywheel: three simulated generations with an improving agent show
   the virtuous cycle quantitatively.

Expected output (seeded, deterministic):
- 200 raw trajectories, ~128 passing
- quality scores form a bell around 0.55-0.65; median ~0.59
- near-dedup finds 3 structural groups where exact dedup kept 38
- repair recovers ~89% of failures via passing siblings
- curriculum phases show rising mean difficulty (0.15 -> 0.34)
- weakness mix overweights testgen, the lowest pass-rate type
- flywheel report shows pass_rate, quality, and yield/raw rising
  across 3 generations

Run: python3 examples/deep_dive.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collector import Collector, ToolCall, Verification
from filter import quality as qf
from filter import scoring, repair
from filter.near_dedup import near_dedup as near_dedup_fn, duplicate_groups
from datasets import curriculum, mixer
from dashboard import flywheel as fw

rng = random.Random(11)

TASK_TYPES = ["bugfix", "feature", "refactor", "testgen"]

PROMPTS = {
    "bugfix": "Fix the off-by-one error in paginate() so the last page is not dropped.",
    "feature": "Add a retry decorator with exponential backoff to the http client.",
    "refactor": "Extract the validation logic from submit() into a standalone function.",
    "testgen": "Write unit tests for the LRU cache covering eviction and TTL expiry.",
}

CODE_TEMPLATES = [
    "def solve_{t}(data):\n    out = []\n    for row in data:\n        if row:\n            out.append(row)\n    return out\n",
    "def solve_{t}(data):\n    out = []\n    for row in data:\n        if row is not None:\n            out.append(row)\n    return out\n",
    # Same structure, different names: near-dup of template 0.
    "def fix_{t}(items):\n    res = []\n    for it in items:\n        if it:\n            res.append(it)\n    return res\n",
    # More complex: branches and nesting.
    "def solve_{t}(data, k=3):\n    out = []\n    for i, row in enumerate(data):\n"
    "        if row and i % k == 0:\n            try:\n                out.append(int(row))\n"
    "            except (TypeError, ValueError):\n                continue\n"
    "        elif not row:\n            out.append(0)\n    return out\n",
]


def synth(collector: Collector, i: int) -> None:
    task_type = TASK_TYPES[i % len(TASK_TYPES)]
    task_id = f"{task_type}-{(i // len(TASK_TYPES)) % 10}"
    traj = collector.open(task_id=task_id, task_type=task_type,
                          prompt=PROMPTS[task_type], model="synth-agent-2")

    n_steps = rng.randint(2, 14)
    for s in range(n_steps):
        obs = "ok"
        r = rng.random()
        if r < 0.15:
            obs = "error: test failed, retrying"
        elif r < 0.20:
            obs = "traceback: assertion error in run_tests"
        traj.steps.append(ToolCall(
            tool=rng.choice(["read", "edit", "run_tests", "search"]),
            arguments={"target": f"module_{s}.py"},
            observation=obs,
            latency_ms=rng.uniform(50, 2000),
        ))

    roll = rng.random()
    # testgen is deliberately weaker: more headroom for the mixer to find.
    pass_p = 0.70 if task_type != "testgen" else 0.45
    if roll < pass_p:
        passed, total = 5, 5
    elif roll < pass_p + 0.12:
        passed, total = rng.randint(3, 4), 5  # near miss
    else:
        passed, total = rng.randint(0, 2), 5

    code = CODE_TEMPLATES[i % len(CODE_TEMPLATES)].format(t=task_id.replace("-", "_"))
    collector.close(
        traj,
        final_code=code,
        verification=Verification(passed=(passed == total),
                                  tests_passed=passed, tests_total=total),
        prompt_tokens=rng.randint(200, 1500),
        completion_tokens=rng.randint(100, 3000),
        total_latency_ms=sum(s.latency_ms for s in traj.steps),
    )


def main() -> None:
    collector = Collector()
    for i in range(200):
        synth(collector, i)
    raw = collector.trajectories
    print(f"Collected {len(raw)} trajectories\n")

    # 1. Quality scoring.
    print("-" * 64)
    print("1. QUALITY SCORING")
    print("-" * 64)
    scores = scoring.score_all(raw)
    print(scoring.render_distribution(scores))
    print("\ntop 3 trajectories:")
    for s in scoring.top_k(scores, 3):
        comp = ", ".join(f"{k}={v:.2f}" for k, v in s.components.items())
        print(f"  {s.trajectory_id} score={s.score:.3f} ({comp})")
    print()

    # 2. Near-dedup vs exact dedup.
    print("-" * 64)
    print("2. NEAR-DEDUP (AST) vs EXACT DEDUP")
    print("-" * 64)
    passes = [t for t in raw if t.verification and t.verification.passed]
    exact = qf.dedup(passes)
    struct = near_dedup_fn(passes)
    print(f"  exact dedup kept:    {len(exact.kept)} "
          f"(removed {exact.removed})")
    print(f"  near dedup kept:     {len(struct.kept)} "
          f"(removed {struct.removed})")
    groups = duplicate_groups(passes)
    print(f"  structural dup groups: {len(groups)}")
    print()

    # 3. Repair.
    print("-" * 64)
    print("3. TRAJECTORY REPAIR")
    print("-" * 64)
    rep = repair.repair_batch(raw)
    n_failed = sum(1 for t in raw if t.verification
                   and not t.verification.passed)
    print(f"  failed trajectories: {n_failed}")
    print(f"  repaired via sibling: {len(rep.repaired)}")
    print(f"  rescued near misses:  {len(rep.near_misses)}")
    print(f"  unrecoverable:        {rep.unrecoverable}")
    recovered = len(rep.repaired) + len(rep.near_misses)
    if n_failed:
        print(f"  recovery rate: {recovered / n_failed:.1%}")
    if rep.repaired:
        r0 = rep.repaired[0]
        print(f"  example: {r0.trajectory_id}: {r0.repair_note}")
    print()

    # Usable pool: passes + repaired.
    usable = passes + rep.repaired

    # 4. Curriculum.
    print("-" * 64)
    print("4. CURRICULUM ORDERING")
    print("-" * 64)
    cur_rows = curriculum.emit_curriculum_sft(usable)
    bins = curriculum.curriculum_bins(cur_rows, n_bins=4)
    for b in curriculum.bin_summary(bins):
        print(f"  phase {b['phase']}: n={b['n']:3d} "
              f"mean_difficulty={b['mean_difficulty']:.3f}")
    if cur_rows:
        easy, hard = cur_rows[0], cur_rows[-1]
        print(f"  easiest: difficulty={easy['difficulty']:.3f} "
              f"({easy['task_type']})")
        print(f"  hardest: difficulty={hard['difficulty']:.3f} "
              f"({hard['task_type']})")
    print()

    # 5. Mixing. Plan on the raw batch so the weakness weights reflect
    # real pass-rate differences; sample from the usable pool.
    print("-" * 64)
    print("5. DATA MIXING (weakness-weighted)")
    print("-" * 64)
    plan = mixer.plan_mix(raw, strategy="weakness")
    print(mixer.render_plan(plan))
    mixed = mixer.apply_mix(usable, plan, scores=scores)
    print(f"  sampled {len(mixed)} trajectories under the plan")
    print()

    # 6. Flywheel over three simulated generations.
    print("-" * 64)
    print("6. FLYWHEEL (3 simulated generations, improving agent)")
    print("-" * 64)
    tracker = fw.FlywheelTracker()
    gen_params = [
        (1, 200, 0.55, 0.52),
        (2, 260, 0.68, 0.61),
        (3, 340, 0.78, 0.70),
    ]
    for gen, n, pr, mq in gen_params:
        sft_y = int(n * pr * 0.92)
        tracker.record(fw.GenerationStats(
            generation=gen, n_raw=n, pass_rate=pr, mean_quality=mq,
            sft_yield=sft_y, pref_yield=int(n * pr * 0.3),
            repaired=int(n * (1 - pr) * 0.5)))
    print(tracker.report())
    print()
    print("Base pipeline (filter -> SFT) is covered by worked_example.py.")


if __name__ == "__main__":
    main()
