# trajectory-to-training guide

For what this repo found and why it matters, see the [README](../README.md) and the [research summary](research.md).

## Pipeline stages

1. `collector.Trajectory` is the schema. One trajectory per task attempt: the prompt, the sequence of tool calls and observations, the final code, and the verification result (tests passed/total). The collector API is two calls: open a trajectory, then close it with the outcome.

2. Each trajectory gets a quality score from its verification outcome, trace structure, and failure signals. Scoring is monotonic and inspectable: you can read a score and know exactly what produced it.

3. `filter.quality` applies gates in order: verification outcome (keep passes; optionally keep instructive failures for preference data), trajectory length bounds (too short is degenerate, too long is rambling), exact deduplication on normalized code, and per-task diversity caps so one easy task does not dominate the dataset. A separate structural pass, `filter.near_dedup`, collapses solutions that share the same code structure even when the text differs.

4. Not every failed trajectory is discarded. When another run on the same task passed, `filter.repair` keeps the failed run's steps up to where it went wrong and continues with the passing sibling's steps and verified code, recording the source. Failures with no passing sibling but a high partial pass rate are kept as near misses for preference data.

5. `datasets.sft` converts passing trajectories to chat-format SFT rows, optionally including the tool-call trace. `datasets.preference` builds chosen/rejected pairs from the same task: a passing trajectory versus a failing one. `datasets.split` does a stratified train/val split by task type.

6. Curriculum ordering sorts the emitted dataset from easy to hard by task type and trajectory complexity, so training starts with the easy cases.

7. Data mixing rebalances the dataset toward the task types where the current model fails most. The mix weights come from the flywheel tracker.

8. `dashboard.metrics` computes verification rate by task type, trajectory length distribution, per-filter pass rates, and SFT/preference yield. Run it after every collection batch to catch data regressions.

## The flywheel

Each training generation records its dataset composition, the model it produced, and that model's verification rate per task type. The next generation's mix weights and curriculum order come from the previous generation's weaknesses. The dashboard shows the trend: is each generation actually getting better, and on what.

## Architecture

```
trajectories (raw agent runs)
    |
collector/trajectory.py   capture tool calls, observations, final code, verification
    |
filter/scoring.py         continuous quality score per trajectory
filter/quality.py         verification gate, length bounds, dedup, diversity caps
filter/near_dedup.py      structural (AST) dedup that catches renamed copies
filter/repair.py          rebuild failures from a verified sibling, or keep near misses
    |
datasets/sft.py           SFT converter, with or without the tool-call trace
datasets/preference.py    chosen/rejected pairs from the same task
datasets/split.py         stratified train/val split
datasets/curriculum.py    easy-to-hard ordering
datasets/mixer.py         weakness-weighted rebalancing
    |
dashboard/metrics.py      yield, verification by task type, length stats
dashboard/flywheel.py     per-generation trends
```

## Design principles

1. Standard library only. No numpy, no pandas, no torch. The pipeline runs on any machine that runs the agent. Dataset files are JSONL.
2. Filters key off the verification outcome, not a learned reward model.
3. Traces are optional in SFT. Including full tool-call traces teaches the model the process; final-code-only teaches the product. The converter supports both so you can ablate.
4. Repair is conservative. A repaired trajectory takes the verified final code of a passing sibling on the same task and records its source. Failures with no passing sibling are kept only as near misses for preference data.

## Worked example

See `examples/worked_example.py`. It synthesizes trajectories across task types with realistic failure modes, runs the full pipeline, and prints the quality report. Expected output is documented at the top of the file. Verified end to end locally.

## Related

Built to pair with [agent-eval-harness](https://github.com/pdj555/agent-eval-harness), which produces the verified trajectories this pipeline consumes. The collector schema is harness-agnostic: any system that can report tool calls and a test outcome can feed this pipeline.

---

Built by [Preston Jones](https://github.com/pdj555).
