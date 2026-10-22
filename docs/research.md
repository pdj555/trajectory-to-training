# Research summary

Coding agents get better by training on their own past runs. Run the agent, keep the runs that worked, train on them, repeat. Whatever counts as "worked" becomes the next model's habits.

This repo is that pipeline. It's part of one question: **how can we trust agents to merge without a human reading every diff?** [agent-harness](https://github.com/pdj555/agent-harness) answers it at merge time. This is where the answer compounds, because a bad check here doesn't just let one change through, it trains the behavior in.

## What it does

It takes raw agent runs (the prompt, every tool call, the final code, and whether the tests passed) and turns them into fine-tuning and preference data. The only label it trusts is the test result. No human annotation, no learned reward model.

Along the way it scores each run, drops near-duplicates by comparing code structure rather than text, repairs failed runs from a passing run on the same task, orders the data easy to hard, and reweights toward task types the model fails most.

On a seeded synthetic batch of 200 runs:

- the worked example keeps 123 fine-tuning rows and 37 preference pairs
- repair recovers 62 of 72 failed runs from a passing sibling, keeps 2 as near misses, and drops 8
- every repaired run records which run it was rebuilt from

```bash
python3 examples/worked_example.py
python3 examples/deep_dive.py
```

## Where it breaks

- **The label is the weak point.** A run that games its tests is labeled a success and goes straight into the training set. agent-harness shows 27 changes that pass every test and shouldn't ship. Through this pipeline, each one would become a fine-tuning example.
- **Preference pairs make it worse.** A gamed run that passes gets paired as "chosen" against an honest run that failed, which teaches the model to prefer the cheat. Among passing runs, the pipeline picks the one with the fewest steps, and a quick hardcoded answer is likely to win that.
- **All data here is synthetic.** The flywheel numbers in the deep dive come from a simulated, improving agent, not a trained model.
- **No model has been trained on the output yet.**

## Next

1. Use the agent-harness spec gate as the label instead of the tests alone, and count how many "passing" runs it rejects.
2. Seed the batch with the 27 agent-harness attacks and measure how many reach the training set with and without the gate.
3. Train a small model on both versions and compare how often it cheats.

Preston Jones · p@pdj.dev
