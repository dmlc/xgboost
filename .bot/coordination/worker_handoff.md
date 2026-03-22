# Worker Handoff — GPT-5.3-Codex

## Task
Handle `dmlc/xgboost` issue **#11845** in this worktree with strict scope control.

This is a feature request for a new hyperparameter that regularizes tree depth by decaying the effective update within a tree as node depth increases.

## First read
Before doing anything else, read:
- `AGENTS.md`
- `.bot/retrieved_memory.md`
- `.bot/memory/worker_retrieved_memory.md` if present
- `.bot/coordination/coordinator_plan.md`

## Workspace
- Worktree: `/workspace/repos/dmlc__xgboost/worktrees/issue-11845-suggestion-for-new-hyperparameter-regula`
- Branch: `bot/issue-11845-suggestion-for-new-hyperparameter-regula`

## Hard guardrails
- **Do not open PRs**
- **Do not push**
- **Do not run dangerous commands**
- Keep scope narrow and additive
- Ignore unrelated untracked workspace/persona files unless explicitly needed

## Important issue context
The issue is **not** a normal bug. It is a proposal for a new hyperparameter.
Maintainers have not clearly asked for an implementation yet; they raised concerns about:
- theoretical justification
- whether existing regularization already covers the use case
- hyperparameter burden and parameterization choices

That means you must not force a code patch if the change is too broad or under-specified.

## Existing code facts you must internalize
### 1) Current eta behavior is post-structure scaling
Current `learning_rate`/`eta` affects leaf outputs after the tree structure is chosen.
It does **not** affect split structure.

See:
- `tests/cpp/tree/test_tree_stat.cc`
  - `TestSplitWithEta`

That test verifies changing eta changes leaf values but preserves split structure.

### 2) Existing callback eta decay is NOT this feature
Do **not** confuse this issue with the existing per-round learning-rate scheduler.

Existing callback paths:
- `python-package/xgboost/callback.py`
- `python-package/xgboost/testing/callbacks.py`
- `tests/python/test_callback.py`
- `tests/python-gpu/test_gpu_callbacks.py`

Those operate **across boosting rounds**, not within a single tree by node depth.

## Decision gate: choose one of two outcomes

### Outcome A — No safe minimal patch
If, after inspecting the relevant tree updaters and evaluator code, you conclude that an upstream-quality implementation would be too invasive or under-specified for a minimal validated patch, **stop there** and write a precise technical note/report.

Your note should explain:
- why a small patch would be misleading or incomplete
- which updaters/evaluators would need coordinated changes
- whether a leaf-scaling-only patch would be semantically insufficient
- what the smallest plausible future design would be

This is a valid outcome.

### Outcome B — Narrow, coherent prototype patch
Only attempt code if you can make it default-compatible and coherent enough to be a real candidate patch.

## If you attempt a patch

### Preferred parameter name
Use:
- `depth_decay`

Avoid a vague name like `decay`.

### Preferred semantics
Use 0-based internal node depth:
- effective per-node scale:
  - `learning_rate * pow(depth_decay, depth)`
- depth 0 (root leaf / stump): unchanged
- depth 1: first decay step

### Default compatibility
- default must be `1.0`
- behavior with `depth_decay=1.0` must match current behavior

### Required source touchpoints to inspect before coding
At minimum inspect these files before deciding patch/no-patch:
- `src/tree/param.h`
- `src/tree/split_evaluator.h`
- `src/tree/updater_colmaker.cc`
- `src/tree/hist/evaluate_splits.h`
- `src/tree/updater_quantile_hist.cc`
- `src/tree/updater_gpu_hist.cu`
- `src/tree/gpu_hist/multi_evaluate_splits.cu`
- `src/tree/gpu_hist/leaf_sum.cu`
- `tests/cpp/tree/test_tree_stat.cc`

### Critical design question
You must decide whether you are implementing:
1. **leaf-output scaling only**, or
2. **split-search-aware depth regularization**

Do not blur them together.

#### If you choose leaf-output scaling only
Be explicit in your report/commit notes that:
- this approximates the proposal
- it does **not** cause deeper nodes to become less likely during split search
- it inherits the current eta-like behavior where structure is selected first

This may still be too weak for the issue.

#### If you choose split-search-aware regularization
Expect more invasive changes. You will need to reason carefully about:
- `CalcWeight(...)`
- `CalcGain(...)`
- evaluator logic
- how depth is threaded through exact / hist / GPU split application and evaluation

Do not claim this path is minimal unless you actually prove it is.

## Minimum acceptable engineering bar for a patch
A patch is only worth keeping if it is coherent across the main tree paths:
- exact (`grow_colmaker`)
- CPU hist / approx
- GPU hist / approx where applicable

A one-updater-only patch is not a strong answer for this issue.

## Suggested implementation route if you proceed
1. Add `depth_decay` to `src/tree/param.h`
   - default `1.0`
   - clear description that values below 1 shrink deeper node outputs
2. Determine where node depth is already available in each updater path
3. Apply consistent per-depth scaling where leaf values are assigned
4. If you also alter split search, update gain/weight logic consistently
5. Add targeted C++ tests
6. Run lint and the strongest feasible targeted tests

## Tests to add or adapt if patching
### Must-have test themes
1. **Default invariance**
   - `depth_decay=1.0` behaves like baseline
2. **Depth effect exists**
   - deeper leaves get smaller outputs when `depth_decay < 1`
3. **Any claimed structure effect is real**
   - only add this if your implementation actually changes split search

### Best starting point
Use the style of:
- `tests/cpp/tree/test_tree_stat.cc`
  - especially `TestSplitWithEta`

That file already encodes current eta invariants across updaters.

## Validation
### Required
Run repo lint:
```bash
pre-commit run --all-files --show-diff-on-failure
```

### Strongly recommended if build/test environment exists
If `build/testxgboost` is already available, run a narrow gtest filter:
```bash
./build/testxgboost --gtest_filter=TestSplitWithEta.*:*DepthDecay*
```

If the binary does not exist but a normal local build is feasible, use the documented C++ unit-test flow from `doc/contrib/unit_tests.rst`.
Do not invent commands if the environment is missing prerequisites.

## What not to do
- Do not spend time on large docs work
- Do not add Python callback APIs for this feature
- Do not write a CPU-only prototype and pretend it is sufficient
- Do not confuse round-wise eta scheduling with depth-wise within-tree decay
- Do not claim theoretical justification you did not establish

## Recommended practical posture
Be conservative.

This issue is acceptable for coding only if you can make a small, honest, validated change.
If not, the correct output is a precise engineering note explaining why no minimal patch is safe.
