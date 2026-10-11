# trajectory-to-training

Turn coding-agent trajectories into training data. Capture raw agent runs, filter them for quality, and emit supervised fine-tuning and preference datasets. Every row carries a ground-truth label: did the code pass the tests. Zero human annotation.

## Quickstart

```bash
# No dependencies beyond Python 3.10+. Everything is stdlib.
python3 examples/worked_example.py
```

That runs a full pass: generates sample trajectories, scores them, filters them, emits SFT and preference datasets, and prints a data-quality report.

## Why this matters

Frontier coding models improve on a data flywheel: deploy the agent, collect trajectories, train on the good ones, deploy the better agent. The bottleneck is not model architecture. It is the system that turns raw interaction logs into clean, verifiable training signal. That is what this repo builds.

## Pipeline stages

1. **Collect.** `collector.Trajectory` is the schema. One trajectory per task attempt: the prompt, the sequence of tool calls and observations, the final code, and the verification result (tests passed/total). The collector API is two calls: open a trajectory, then close it with the outcome.

2. **Score.** Continuous trajectory quality scoring. Each trajectory gets a quality score from its verification outcome, trace structure, and failure signals. Scoring is monotonic and inspectable: you can read a score and know exactly what produced it.

3. **Filter.** `filter.quality` applies gates in order: verification outcome (keep passes; optionally keep instructive failures for preference data), trajectory length bounds (too short is degenerate, too long is rambling), structural near-deduplication (not just exact-match; structurally identical solutions collapse), and per-task diversity caps so one easy task does not dominate the dataset.

4. **Repair.** Failed trajectories are not all trash. The repair stage identifies failures that are one edit away from passing, a wrong constant, a missing import, an off-by-one, and salvages them into the training set with the fix recorded. Repair is conservative: it only touches failures it can verify.

5. **Emit.** `datasets.sft` converts passing trajectories to chat-format SFT rows, optionally including the tool-call trace. `datasets.preference` builds chosen/rejected pairs from the same task: a passing trajectory versus a failing one. `datasets.split` does a stratified train/val split by task type.

6. **Order.** Curriculum ordering sorts the emitted dataset from easy to hard by task type and trajectory complexity, so training starts on clean wins and ramps to the hard cases.

7. **Mix.** Weakness-weighted data mixing rebalances the dataset toward the task types where the current model fails most. The mix weights come from the flywheel tracker, not from vibes.

8. **Measure.** `dashboard.metrics` computes the numbers that matter: verification rate by task type, trajectory length distribution, per-filter pass rates, and SFT/preference yield. Run it after every collection batch to catch data regressions.

## The flywheel

Multi-generation tracking closes the loop. Each training generation records its dataset composition, the model it produced, and that model's verification rate per task type. The next generation's mix weights and curriculum order come from the previous generation's weaknesses. The dashboard shows the trend: is each generation actually getting better, and on what.

## Architecture

```
trajectories (raw agent runs)
    |
collector/      capture tool calls, observations, final code, verification
    |
scoring/        continuous quality score per trajectory
    |
filter/         verification gates, length bounds, structural dedup, diversity caps
    |
repair/         salvage near-miss failures with verified fixes
    |
datasets/       SFT converter, preference-pair builder, train/val split
    |
curriculum/     easy-to-hard ordering
mixing/         weakness-weighted rebalancing
    |
dashboard/      yield rate, failure-mode breakdown, flywheel trends
```

## Design principles

1. **Stdlib only.** No numpy, no pandas, no torch. The pipeline runs on any machine that runs the agent. Dataset files are JSONL.
2. **Verification is the label.** Filters key off the verification outcome, not a learned reward model. Ground truth beats vibes.
3. **Traces are optional in SFT.** Including full tool-call traces teaches the model the process; final-code-only teaches the product. The converter supports both so you can ablate.
4. **Repair is conservative.** A salvaged trajectory must pass verification after repair, or it stays out.

## Worked example

See `examples/worked_example.py`. It synthesizes trajectories across task types with realistic failure modes, runs the full pipeline, and prints the quality report. Expected output is documented at the top of the file. Verified end to end locally.

## Related

Built to pair with [agent-eval-harness](https://github.com/pdj555/agent-eval-harness), which produces the verified trajectories this pipeline consumes. The collector schema is harness-agnostic: any system that can report tool calls and a test outcome can feed this pipeline.

---

Built by [Preston Jones](https://github.com/pdj555). Part of a series on agent infrastructure: evals, training data, and failure analysis.
