# Coordinator Plan — dmlc/xgboost issue #11845

## Mission
Prepare a GPT-5.3-Codex worker to handle issue **#11845: "Suggestion for new hyperparameter: Regularize tree depth"** with high precision and strong scope control.

This is **not** a routine bug fix. It is a feature suggestion that touches core tree-building behavior and has not been clearly accepted by maintainers yet.

## Inputs reviewed
- `AGENTS.md`
- `.bot/retrieved_memory.md`
- `.bot/memory/coordinator_retrieved_memory.md`
- GitHub issue body and comments for `dmlc/xgboost#11845`
- Relevant source/test files in this worktree

## What the issue thread says
### Request from the issue author
The author proposes a new hyperparameter that decays the effective learning rate **within a tree by depth**, e.g. deeper splits/leaves get smaller updates.

### Maintainer signals
The maintainer discussion is exploratory, not an implementation request. The main concerns raised are:
- whether existing regularization (`min_split_loss`, `min_child_weight`) already captures the effect
- lack of a stronger theoretical justification
- uncertainty about the exact decay form / parameterization
- concern that this adds another hyperparameter burden for users

### Practical implication
A worker should **not assume** there is an obviously acceptable upstream patch. The worker must first assess whether a minimal, validated, upstream-quality change exists.

## Technical reality in the codebase
### Important existing invariant
In current XGBoost tree boosters, `learning_rate` scales leaf outputs **after** tree structure is chosen. Existing C++ tests explicitly verify that changing `eta` changes leaf weights but **does not change split structure**:
- `tests/cpp/tree/test_tree_stat.cc`
  - `TestSplitWithEta`

This matters because the requested feature is not just “another eta.” A depth-aware penalty could mean either:
1. **leaf-output scaling only** by depth, or
2. **true depth regularization** that also changes split selection / gain ranking at deeper levels.

These are not the same thing.

### Why this feature is risky
A naive patch that only scales final leaf outputs by depth is relatively easy, but it likely **does not fully implement** the requested “regularize tree depth” behavior, because split search/gain evaluation would still favor deeper nodes exactly as before.

A more faithful implementation would need coordinated changes across multiple updaters and evaluators.

## Concrete code touchpoints
### Parameter definition
- `src/tree/param.h`
  - `TrainParam`
  - `DMLC_DECLARE_FIELD(...)`
  - `CalcWeight(...)`
  - `CalcGain(...)`

### Split evaluation / tree growth
- `src/tree/split_evaluator.h`
- `src/tree/updater_colmaker.cc`            # exact
- `src/tree/hist/evaluate_splits.h`         # CPU hist split application
- `src/tree/updater_quantile_hist.cc`       # hist / approx / multi-target paths
- `src/tree/updater_gpu_hist.cu`            # GPU hist / approx core split application
- `src/tree/gpu_hist/multi_evaluate_splits.cu`
- `src/tree/gpu_hist/leaf_sum.cu`

### Existing nearby tests
- `tests/cpp/tree/test_tree_stat.cc`
- `tests/cpp/tree/hist/test_evaluate_splits.cc`
- `tests/cpp/tree/gpu_hist/test_evaluate_splits.cu`
- Python callback tests that mention eta decay are **not** the same feature:
  - `python-package/xgboost/testing/callbacks.py`
  - `tests/python/test_callback.py`
  - `tests/python-gpu/test_gpu_callbacks.py`

## Key insight for the worker
Do **not** confuse the existing per-boosting-round `LearningRateScheduler` callback with this issue.

Existing callback-based eta decay changes learning rate **across rounds/trees**.
This issue asks for decay **within a single tree by node depth**.

## Recommended worker strategy

### Primary recommendation
Treat this issue as a **feasibility-gated implementation task**.

The worker has two acceptable outcomes:

#### Outcome A — No safe minimal code patch
If, after inspecting the touchpoints above, the worker concludes that a small upstream-quality patch is not credible without broad algorithmic work, the worker should **not force a patch**. Instead it should produce a precise technical note/report describing:
- why the issue is under-specified or too invasive for a safe minimal patch
- which components would need coordinated changes
- what the smallest plausible design would be if the maintainers want to pursue it

#### Outcome B — Narrow, coherent prototype patch
Only if the worker can keep behavior coherent and well-tested should it implement a patch.

## If the worker attempts a patch
### Preferred parameter name
Prefer a clear name like:
- `depth_decay`

Avoid a vague name like just `decay`.

### Preferred semantics
If implemented, define effective per-node scaling using **0-based internal depth**:
- effective learning rate at node depth `d`:
  - `learning_rate * pow(depth_decay, d)`
- root leaf/stump at depth `0` keeps the base learning rate
- children of the root (depth `1`) get the first decay factor

This matches the issue author's example if interpreted against internal tree depth.

### Default / compatibility
- default must be `1.0`
- default behavior must remain equivalent to current behavior
- no user-visible behavior change when `depth_decay=1`

### Scope bar for an acceptable patch
A patch is only credible if it is applied coherently to the major tree-growing paths used in this repo:
- exact (`grow_colmaker`)
- approx / hist CPU paths
- GPU hist / GPU approx path where relevant shared logic exists

A CPU-only or single-updater prototype is **not** a strong upstream-quality fix for this issue.

### Design decision the worker must make explicitly
Before coding, decide whether the patch is:
1. **leaf-output scaling only**, or
2. **split-search-aware depth regularization**

If it is only (1), the worker must explicitly acknowledge that this is a limited approximation and may not satisfy the feature request fully.

## Tests/validation the worker should target
### Required repo-wide validation
- `pre-commit run --all-files --show-diff-on-failure`

### Strongly recommended targeted C++ validation
If a build environment is available:
- use the monolithic gtest binary (`testxgboost`) and run targeted filters around:
  - existing `TestSplitWithEta`
  - any new `DepthDecay` tests

Potential command pattern if `build/testxgboost` exists:
- `./build/testxgboost --gtest_filter=TestSplitWithEta.*:*DepthDecay*`

If build/test infrastructure is not present, the worker should say so explicitly rather than invent commands.

## What a good minimal test looks like
If a code patch is attempted, add tests that prove:
1. `depth_decay=1.0` preserves current behavior
2. leaf outputs at deeper nodes are reduced when `depth_decay < 1`
3. any claimed effect on split selection is actually validated, not assumed

The existing `TestSplitWithEta` is the best starting reference for how eta-related invariants are tested here.

## Explicit non-goals
Do not spend time on these unless the patch genuinely lands:
- Python/R docs sweep
- large benchmark suite reproduction from the issue comment
- new callbacks
- user-facing tutorial work
- maintainers' theory discussion beyond short implementation notes
- PR creation or external posting

## Coordinator guidance summary
The worker should be conservative.

This issue is more like:
- “research proposal + core algorithm change”
than
- “localized bug fix”.

So the worker should only implement if it can do it in a way that is:
- default-compatible
- coherent across updaters
- narrowly tested
- honest about semantics

Otherwise, the correct outcome is a precise no-patch technical note, not a forced half-implementation.
