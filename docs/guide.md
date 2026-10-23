# trajectory-to-training guide

What this repo found is in the [README](../README.md). The experiment, in full, is in the [research summary](research.md). This guide covers the modules: what each one takes, what it returns, and the rules it follows.

## The record: collector/trajectory.py

A trajectory is one attempt at one task: task id, a task type string, the prompt, the steps, the final code, and a verification record (passed, tests passed, tests total). Steps are tool calls: tool name, arguments, observation, latency. A repaired flag and repair note are also on the record; only `filter/repair.py` sets them.

The Collector API is two calls. `open(task_id, task_type, prompt)` starts a record. `close(traj, final_code, verification)` finishes it and stores it. `save_jsonl` and `load_jsonl` move records in and out as JSONL.

`Trajectory.code_hash()` is the key the exact dedup pass uses: SHA 256, first 16 hex characters, over the code with blank lines and comment only lines removed and trailing spaces stripped.

## Scoring: filter/scoring.py

`score_all` gives every trajectory a score from 0 to 1. It is a weight, not a gate: nothing is dropped for a low score. Four parts, weighted:

- Verification, 0.40: the test pass rate, shrunk toward zero when the suite is thin. The shrink is tests total divided by tests total plus 3, so a pass on 1 test counts for less than a pass on 12.
- Elegance, 0.25: branch count, statement nesting depth, and line count from the parsed code. More of each lowers the score. Code that does not parse gets 0.3 for this part.
- Efficiency, 0.20: step count against the median step count for the task type. At or under the median scores 1.0. Steps whose observation mentions an error cut the score.
- Diversity, 0.15: rare task types score higher. A run whose code hash already appeared in the batch gets 0.1 on the uniqueness half.

Scores are batch relative. Medians, type counts, and seen hashes all come from the batch being scored, so the same run can score differently in a different batch.

## Gates: filter/quality.py

`run_all` applies four gates in order. Each gate returns the survivors and a count of what it removed.

1. Verification. Keeps runs that passed. With `keep_failures=True` it keeps failures too, which is what the preference path uses.
2. Length. Keeps runs with 1 to 50 steps by default.
3. Exact dedup. Drops repeated code hashes, keeping the first.
4. Diversity cap. At most 10 runs per task by default. Within a task, passing runs come first, then higher pass rate, then fewer steps.

## Structural dedup: filter/near_dedup.py

Exact dedup misses the common copy: the same solution with new names. `near_dedup` parses the code, rewrites every name, argument, function, class, and attribute to a fixed placeholder, and dedups on that structure. Code that does not parse falls back to the exact hash. `duplicate_groups` returns the groups of two or more, for inspection.

## The second check: filter/second_check.py

`relabel(trajectories, check)` is the module the finding rests on. `check` is any function that takes a trajectory and returns (ok, reason).

- Only runs that passed their tests are checked. Runs that failed, and runs with no verification record, pass through unchanged.
- When the check fails a run, `relabel` returns a copy with passed set to false. The copy keeps the original tests passed and tests total counts, so the record still shows that the tests passed.
- The originals are never modified. The report lists how many runs were checked, which ids flipped, and the reason string the check returned for each.

What a flipped run does next: it leaves the fine tuning rows, and in preference building it counts as a failure. One detail matters. It keeps a pass rate of 1.0, and the pair builder picks the lowest pass rate as the rejected side, so a flipped run is usually not picked as the rejected side either. It is removed, not used as a negative example.

The check used in the experiment is `hidden_input_check` in `examples/gamed_runs.py`. It runs the run code and a reference solution on inputs the tests never use and requires an exact match on all of them.

## Repair: filter/repair.py

`repair_batch` groups runs by task and handles each failure one of three ways.

- A passing run exists on the task: the failure is repaired. `repair_from_sibling` finds the failure point, the first step whose observation mentions an error, a traceback, a failure, an exception, or a timeout. It keeps the failed run steps before that point, then appends the full step list and the final code of the best passing run on the task: highest pass rate, then fewest steps. The new record is marked repaired, its id gains a "-repaired" suffix, and its repair note names the source run and how many prefix steps were kept.
- No passing run, but the failure passed at least 0.6 of its tests: kept as a near miss for preference data.
- Otherwise: counted as unrecoverable and dropped.

One limit inside the mechanism. A run that failed with no error in any step observation has no failure point, so it keeps all of its steps as the prefix. A repaired record can carry the full failed attempt followed by the full passing one.

Repair never writes new code. The final code in a repaired record is the passing run code, word for word.

## Datasets: datasets/

`sft.py`: `emit_sft` turns passing runs into rows of `{"messages": [user prompt, assistant answer], "task_id", "task_type"}`. The assistant answer is the final code, or, with `include_trace=True`, the step list followed by the final code.

`preference.py`: `build_pairs` makes at most one pair per task, from at least one pass and one failure. "chosen" is the passing run with the highest pass rate, ties broken by fewest steps. "rejected" is the failing run with the lowest pass rate, ties broken by most steps. The pair carries both pass rates.

`split.py`: `stratified_split` groups rows by task type, shuffles with a fixed seed (42), and holds out 20 percent of each type for validation by default. A type with a single row sends nothing to validation.

`curriculum.py`: difficulty scores each run from 0 to 1: trajectory length against a 20 step cap (0.35), the task failure rate in the batch (0.35), and branch count against a cap of 8 (0.30). `emit_curriculum_sft` returns passing runs as SFT rows ordered easy to hard. `curriculum_bins` cuts the ordered rows into phases.

`mixer.py`: `plan_mix` allocates rows across task types. Strategies: uniform (equal shares), weakness (share grows as the type pass rate falls, with a floor so no type vanishes), target (shares the caller supplies). `apply_mix` takes the allocated count per type, best rows first. It never invents rows: a type with too few runs gives what it has, and the shortfall goes to types with room.

## Dashboard: dashboard/

`metrics.py` builds a text report: verification rate by task type, step and token percentiles (p50, p90), the kept and removed count for each gate, and the SFT and preference yields.

`flywheel.py` is a record format, not a measurement. `FlywheelTracker` stores one `GenerationStats` per training generation and reports the change between generations. Nothing in this repo trains a model. The three generations in `examples/deep_dive.py` are numbers written into the script, and the report only does arithmetic on them.

## Examples

- `examples/gamed_runs.py`: the experiment. 14 hand written runs, real tests run in a separate interpreter, the pipeline run twice. Start here.
- `examples/worked_example.py`: 200 synthetic runs through collect, filter, emit, split, and report. Writes `sft_example.jsonl` and `pref_example.jsonl` next to the script.
- `examples/deep_dive.py`: the same kind of batch, showing scoring, structural dedup, repair, curriculum, mixing, and the flywheel record on simulated generations.

Expected output for the two synthetic examples is written at the top of each file.

## Design rules

1. Standard library only. No numpy, no pandas, no torch. Dataset files are JSONL.
2. Filters key off the verification outcome, not a learned reward model.
3. Traces are optional in SFT rows. Code only rows teach the answer. Trace rows teach the steps. The converter supports both so the two can be compared.
4. Repair is conservative. The corrected code always comes from a verified passing run on the same task, and the source is recorded.

## Related

[agent-eval-harness](https://github.com/pdj555/agent-eval-harness) produces verified runs of the kind this pipeline consumes. The collector record is not tied to it. Any system that can report tool calls and a test outcome can feed this pipeline.

---

Built by [Preston Jones](https://github.com/pdj555).
