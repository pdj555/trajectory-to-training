# trajectory-to-training

when passing tests was the only label for good training data, 4 of 10 training rows came from runs that had gamed their tests, because training on agent traces without checking how they passed teaches the next model to cheat

This repo turns runs from coding agents into training data. Each run is labeled by its test result, and this repo adds one more check on the runs that passed before they become training rows.

## The numbers

From `examples/gamed_runs.py`, 14 runs on 2 tasks, 7 attempts per task:

- 10 of the 14 runs passed their tests. 4 of those 10 had gamed the tests: they fit the tested calls instead of solving the task.
- Labeled by tests alone, the 10 passing runs became 10 training rows, 4 of them from gamed runs. Both preference pairs picked a gamed run as the chosen side.
- A second check compared every passing run with a reference solution on inputs the tests never use. It marked all 4 gamed runs as failed and flipped no honest run. On one task the gamed runs were wrong on 631 and 427 of 960 hidden inputs. On the other they were wrong on 104 and 106 of 120.
- With the second check, 6 training rows remained, 0 from gamed runs, and no preference pair picked a gamed run.

## How it works

1. `collector` records one record per attempt: the prompt, the tool calls, the final code, and the test result.
2. Each run gets a quality score. Gates drop runs that failed, runs outside the length bounds, and duplicates, exact and structural (same code shape, different names), and cap the runs per task.
3. `filter/second_check.py` runs a check the caller supplies on every run that passed its tests. A run the check fails is marked as failed. The original run is not changed.
4. Runs still marked as passing become training rows. Each task with both a pass and a failure gives one preference pair: the best passing run as chosen, the worst failing run as rejected.
5. A failed run can be rebuilt from a passing run on the same task, with the source run recorded (`filter/repair.py`). Rows can then be split by task type, ordered easy to hard, and mixed toward the task types with the lowest pass rates.

## Run it

Python 3.10 or later, standard library only.

```bash
python3 examples/gamed_runs.py       # the numbers above, end to end
python3 examples/worked_example.py   # the full pipeline on 200 synthetic runs: 123 training rows, 37 preference pairs
python3 examples/deep_dive.py        # scoring, structural dedup, repair, curriculum, mixing
python3 -m pytest tests/             # 3 tests, including the second check on the real batch
```

In `examples/deep_dive.py`, repair rebuilt 62 of 72 failed synthetic runs from a passing run on the same task and kept 2 more as near misses.

## Limits

- The 14 runs are small and were written by hand for this experiment. Both tasks were picked because their tests are easy to game.
- The 200 run batches in the other two examples are synthetic. Their numbers describe the pipeline, not real agents.
- The second check needs a reference solution, or some other check the caller trusts, and most real tasks do not ship one. It trusts that check completely, and nothing in the code guards against a wrong reference.
- The check compares outputs for exact equality. A run that is right but returns its answer in a different shape fails the check.
- No model has been trained on the output yet. Nothing here shows that cleaner data trains a better model.
