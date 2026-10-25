"""The default labeling path runs the second check inside the gate chain.

Before this, a caller had to run the gates and then relabel as a separate
step; labeling by the gate chain alone admitted every gamed run. Now
``run_all`` takes the check and applies it as its first stage, so one call
down the default path admits 0 of the 10 test-passing rows from gamed runs.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from datasets import preference, sft  # noqa: E402
from filter import quality as qf  # noqa: E402
from gamed_runs import build_batch, hidden_input_check  # noqa: E402


def test_default_path_admits_no_gamed_rows():
    batch, gamed = build_batch()
    kept, results = qf.run_all(batch, check=hidden_input_check)
    stage = results[0]
    assert stage.name == "second_check"
    assert stage.removed == len(gamed) == 4
    assert set(stage.detail) == gamed  # every gamed run, no honest run
    rows = sft.emit_sft(kept)
    assert len(rows) == 6
    assert not [t for t in kept if t.trajectory_id in gamed]


def test_default_path_pairs_pick_no_gamed_run():
    batch, gamed = build_batch()
    kept, _ = qf.run_all(batch, keep_failures=True, check=hidden_input_check)
    flipped = [t for t in kept if t.trajectory_id in gamed]
    assert len(flipped) == 4
    assert all(not t.verification.passed for t in flipped)
    pairs = preference.build_pairs(kept)
    code_to_id = {t.final_code: t.trajectory_id for t in batch}
    assert len(pairs) == 2
    assert all(code_to_id[p["chosen"]] not in gamed for p in pairs)
    assert all(code_to_id[p["rejected"]] not in gamed for p in pairs)


def test_check_off_still_admits_gamed_rows():
    batch, gamed = build_batch()
    kept, results = qf.run_all(batch)
    assert all(r.name != "second_check" for r in results)
    assert sum(1 for t in kept if t.trajectory_id in gamed) == 4
    assert len(sft.emit_sft(kept)) == 10


def test_default_path_leaves_originals_alone():
    batch, gamed = build_batch()
    qf.run_all(batch, check=hidden_input_check)
    assert all(t.verification.passed for t in batch if t.trajectory_id in gamed)
