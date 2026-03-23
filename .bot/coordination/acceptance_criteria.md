# Acceptance Criteria — dmlc/xgboost issue #11845

This issue has **two acceptable end states**.
A half-implemented prototype that pretends to be upstream-ready is **not** acceptable.

---

## Acceptable Outcome A — No-patch technical note

### Scope / discipline
- [ ] Worker inspected the issue thread and the core tree touchpoints before deciding not to patch
- [ ] Worker did not open PRs, push branches, or run dangerous commands
- [ ] Worker did not make unrelated cleanup changes

### Technical note quality
- [ ] The report clearly explains why no small, validated, upstream-quality patch is currently safe
- [ ] The report explicitly distinguishes:
  - [ ] leaf-output scaling by depth
  - [ ] split-search-aware depth regularization
- [ ] The report names the concrete code areas that would need coordinated changes, including at least:
  - [ ] `src/tree/param.h`
  - [ ] `src/tree/split_evaluator.h`
  - [ ] `src/tree/updater_colmaker.cc`
  - [ ] `src/tree/updater_quantile_hist.cc`
  - [ ] `src/tree/updater_gpu_hist.cu`
- [ ] The report notes that existing per-round eta callbacks are not the requested feature
- [ ] The report references the existing eta invariant test in `tests/cpp/tree/test_tree_stat.cc`

### Validation / evidence
- [ ] At minimum, repo lint state is not worsened
- [ ] If any code was touched and kept, it is validated as strongly as the environment allows

---

## Acceptable Outcome B — Narrow, coherent code patch

### Parameter/API
- [ ] A clearly named parameter is used (prefer `depth_decay`)
- [ ] Default value is `1.0`
- [ ] Default behavior matches current behavior
- [ ] Parameter description is clear and not misleading

### Semantics
- [ ] The implementation clearly defines what “depth” means (0-based internal node depth is acceptable)
- [ ] The worker explicitly documents whether the patch is:
  - [ ] leaf-output scaling only, or
  - [ ] split-search-aware depth regularization
- [ ] The implementation does not silently claim stronger semantics than it actually provides

### Updater coherence
A kept patch must be coherent across the major tree paths, not just one isolated codepath.

- [ ] Exact tree path considered (`grow_colmaker`)
- [ ] CPU hist / approx path considered
- [ ] GPU hist / approx path considered where the touched logic applies
- [ ] Any intentionally unsupported path is explicitly called out and justified

### Tests
- [ ] There is a targeted test proving `depth_decay=1.0` preserves baseline behavior
- [ ] There is a targeted test proving deeper nodes/leaves receive smaller outputs when `depth_decay < 1`
- [ ] If the patch claims split-search effects, there is a test that actually verifies changed structure/gain behavior
- [ ] Existing eta-related invariants/tests are not accidentally broken without explanation

### Validation
#### Required
- [ ] `pre-commit run --all-files --show-diff-on-failure`

#### Strongly preferred when environment supports it
- [ ] targeted `testxgboost` / gtest execution around eta/depth-decay tests
- [ ] any new or adapted C++ tests pass

### Quality bar
- [ ] Patch is small and reviewable
- [ ] Patch is honest about limitations
- [ ] Patch does not introduce unrelated refactors
- [ ] Patch does not confuse round-wise eta scheduling with depth-wise within-tree decay

---

## Explicitly unacceptable outcomes
- [ ] CPU-only prototype presented as a complete fix
- [ ] Patch that only changes one updater with no justification
- [ ] Patch that adds a vague `decay` parameter without clearly scoped semantics
- [ ] Patch that worsens lint status
- [ ] Patch that claims to regularize depth while only applying unrelated round-wise learning-rate scheduling
