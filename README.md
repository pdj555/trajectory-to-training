# trajectory-to-training

**Turns coding-agent runs into training data, and keeps the runs that cheated out of it.**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.svg">
  <img alt="Ten fine-tuning rows per row of squares. With tests as the label, 4 of 10 came from runs that gamed their tests. With a second check, 0. Preference pairs choosing a cheating run: 2 of 2, then 0 of 2." src="docs/assets/hero-light.svg" width="100%">
</picture>

<sub>One question, four repos: **how can we trust agents to merge without a human reading every diff?**<br>
[agent-harness](https://github.com/pdj555/agent-harness) · [agent-eval-harness](https://github.com/pdj555/agent-eval-harness) · **trajectory-to-training** · [hard-rag-evals](https://github.com/pdj555/hard-rag-evals)</sub>

## The finding

What passes the check becomes the next model. Coding agents improve by training on their own runs, and the usual label is "did the tests pass." A run that games its tests gets that label too.

On 14 real runs across two tasks, verified by running their tests:

| Label | Fine-tuning rows | From gamed runs | Gamed run as "chosen" |
| --- | --- | --- | --- |
| Tests alone | 10 | **4** | **2 of 2 pairs** |
| Tests plus a second check | 6 | 0 | 0 of 2 pairs |

The second check compares each run with a reference on inputs the tests never use. It caught all 4 gamed runs and flipped no honest one. The "chosen" column depends on the gamed runs taking fewer steps, which the preference builder uses to break ties.

## Run it

Python 3.10+, standard library only.

```bash
python3 examples/gamed_runs.py       # gamed runs, with and without a second check
python3 examples/worked_example.py   # collect, filter, emit SFT and preference data
python3 examples/deep_dive.py        # scoring, structural dedup, repair, curriculum, mixing
```

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/flow-dark.svg">
  <img alt="Agent runs are labeled by their tests, then by a second check on hidden inputs. Runs that pass both become fine-tuning rows and chosen preference samples; runs that fail the second check go to the rejected side. 4 gamed runs pass the tests." src="docs/assets/flow-light.svg" width="100%">
</picture>

1. **Collect.** One trajectory per attempt: prompt, tool calls, final code, test result.
2. **Filter.** Verification gate, length bounds, structural dedup, per-task caps.
3. **Check again.** `filter/second_check.py` relabels runs that pass their tests but fail an independent check.
4. **Emit.** Fine-tuning rows, preference pairs from the same task, and a stratified split.
5. **Order and mix.** Easy to hard, weighted toward the task types the model fails most.

## Limits

- The second check needs a reference, and most real tasks don't ship one.
- 14 hand-written runs. The deep dive's 200 runs are synthetic.
- No model has been trained on the output yet.

## Docs

[Research summary](docs/research.md) · [Guide](docs/guide.md)
