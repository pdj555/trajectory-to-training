# trajectory-to-training

Turn coding-agent trajectories into training data.

Coding agents generate enormous volumes of interaction data: tool calls, observations, intermediate reasoning, final code, and verification outcomes. Most of it is thrown away. This pipeline captures it, filters it for quality, and emits supervised fine-tuning and preference datasets that can be used to train the next generation of coding models.

## Why this matters

Frontier coding models improve on a data flywheel: deploy the agent, collect trajectories, train on the good ones, deploy the better agent. The bottleneck is not model architecture. It is the system that turns raw interaction logs into clean, verifiable training signal. That is what this repo builds.

Verification is the key. Every trajectory in this pipeline carries a ground-truth outcome: did the final code pass the tests. That turns noisy agent logs into labeled data with zero human annotation.

## Quickstart

```bash
# No dependencies beyond Python 3.10+. Everything is stdlib.
python3 examples/worked_example.py
```

That runs a full pass: generates sample trajectories, filters them, emits SFT and preference datasets, and prints a data-quality report.

## Architecture

```
trajectories (raw agent runs)
    |
collector/      capture tool calls, observations, final code, verification
    |
filter/         verification outcome, length bounds, dedup, diversity caps
    |
datasets/       SFT converter, preference-pair builder, train/val split
    |
dashboard/      yield rate, failure-mode breakdown, filter pass rates
```

## Pipeline stages

1. **Collect.** `collector.Trajectory` is the schema. One trajectory per task attempt: the prompt, the sequence of tool calls and observations, the final code, and the verification result (tests passed/total). The collector API is two calls: open a trajectory, then close it with the outcome.

2. **Filter.** `filter.quality` applies four gates in order: verification outcome (keep passes; optionally keep instructive failures for preference data), trajectory length bounds (too short is degenerate, too long is rambling), exact dedup on normalized code, and per-task diversity caps so one easy task does not dominate the dataset.

3. **Emit.** `datasets.sft` converts passing trajectories to chat-format SFT rows, optionally including the tool-call trace. `datasets.preference` builds chosen/rejected pairs from the same task: a passing trajectory versus a failing one. `datasets.split` does a stratified train/val split by task type.

4. **Measure.** `dashboard.metrics` computes the numbers that matter: verification rate by task type, trajectory length distribution, per-filter pass rates, and SFT/preference yield. Run it after every collection batch to catch data regressions.

## Design notes

- **Stdlib only.** No numpy, no pandas, no torch. The pipeline should run on any machine that runs the agent. Dataset files are JSONL.
- **Verification is the label.** Filters key off the verification outcome, not a learned reward model. Ground truth beats vibes.
- **Traces are optional in SFT.** Including full tool-call traces teaches the model the process; final-code-only teaches the product. The converter supports both so you can ablate.

## Worked example

See `examples/worked_example.py`. It synthesizes 200 trajectories across 4 task types with realistic failure modes, runs the full pipeline, and prints the quality report. Expected output is documented at the top of the file.

## Related

Built to pair with an agent harness that produces verified trajectories. The collector schema is harness-agnostic: any system that can report tool calls and a test outcome can feed this pipeline.
