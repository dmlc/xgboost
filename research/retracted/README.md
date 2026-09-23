# Retracted benchmark output — do not cite

Everything in this directory is kept only as a record of what went wrong. **No number here
may be used in any report, comparison, or claim about `multi_hessian=exact`.**

## Why it was retracted

Two independent defects, either one sufficient to invalidate the results.

### 1. Test-set selection (methodological, fatal)

`benchmark_exact_vs_diagonal.py` chose the winning hyper-parameter configuration with

```python
if best is None or r["test_mlogloss"] < best["test_mlogloss"]:
```

That selects on the **test** set. Every headline test metric it printed was therefore
optimistically biased by test-set peeking, by an unknown and not necessarily equal amount for
the two modes. The script's own docstring claimed selection was on validation; the code did
not do that.

The same defect reached the derived metrics: `rounds_to_reach` measured reachability on test
predictions against a target that was itself diagonal's mean **test** loss. So every
"time to diagonal quality" figure was a test-derived quantity.

### 2. Concurrent process contamination (measurement)

An earlier `nohup` run believed to have died was in fact alive and executing the same
benchmark simultaneously. Both processes ran `covertype` at `nthread=8` on a 16-thread
machine at the same time. The identical fit was recorded as 1.7s by one process and 1.2s by
the other; the exact/diagonal ratio as 1.75x and 2.15x.

Contaminated timings were **discarded, not corrected**. Rescaling them by an inferred
contention factor would have been a fabricated measurement.

## What was *not* affected

The quality metrics were bit-identical across the two concurrent runs (mlogloss 0.11356 and
0.11273, rounds 300 and 246). That is an unplanned confirmation that training is
deterministic and that the contamination was purely a timing problem. It does not rescue the
numbers, because defect 1 applies to them regardless.

## Replacement

`research/benchmark_v2.py` implements the corrected protocol, in which the search phase never
constructs a test DMatrix, so test leakage is structurally impossible rather than merely
avoided. `research/summarize_v2.py` derives every table from the raw JSONL alone.

## Contents

| File | What it is |
|---|---|
| `RETRACTED_run_log.txt` | stdout of the last contaminated run, kept for debugging only |

The raw JSONL from that run does not exist: it was written only at the end of `main()`, which
never completed.
