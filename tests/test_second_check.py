"""Gamed runs reach the training set on test labels alone, and not after a second check."""

from __future__ import annotations

import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from collector import Collector, Verification  # noqa: E402
from filter.second_check import relabel  # noqa: E402
from gamed_runs import build_batch, hidden_input_check, summarize  # noqa: E402


def test_every_gamed_run_passes_its_tests():
    batch, gamed = build_batch()
    assert gamed and all(t.verification.passed for t in batch if t.trajectory_id in gamed)


def test_gamed_runs_reach_sft_on_test_labels_and_not_after_the_check():
    batch, gamed = build_batch()
    before = summarize("tests", batch, gamed)
    assert before["gamed_sft"] == len(gamed)
    report = relabel(batch, hidden_input_check)
    assert set(report.flipped) == gamed  # every gamed run caught, no honest run flipped
    after = summarize("checked", report.trajectories, gamed)
    assert after["gamed_sft"] == 0 and after["gamed_chosen"] == 0


def test_relabel_leaves_failures_and_originals_alone():
    c = Collector()
    ok = c.close(c.open("t", "feature", "p"), "x", Verification(True, 1, 1))
    bad = c.close(c.open("t", "feature", "p"), "y", Verification(False, 0, 1))
    report = relabel([ok, bad], lambda t: (False, "nope"))
    assert report.checked == 1 and report.flipped == [ok.trajectory_id]
    assert ok.verification.passed  # original untouched
    assert report.trajectories[1] is bad
