# User Guide: Upstreaming fed_secure Branch to XGBoost Main

## Overview

This guide provides step-by-step instructions for upstreaming the `fed_secure` branch to the main XGBoost repository. The branch contains significant federated learning functionality with ~8,000 insertions and ~2,700 deletions across 221 files.

**Branch Summary:**
- **Feature:** Secure federated learning for both horizontal and vertical federated schemes with GPU computation support
- **Scope:** 14 commits, 221 files changed
- **Key Components:** Federated plugin infrastructure, histogram synchronization, secure aggregation, GPU-accelerated federated training

---

## Pre-Submission Checklist

### 1. Code Quality and Testing

#### Run All Linters Locally

Before submitting, ensure all code passes style checks:

**Python:**
```bash
cd /path/to/xgboost/
python ./tests/ci_build/lint_python.py --fix
```

**C++:**
```bash
cd /path/to/xgboost/
python ./tests/ci_build/lint_cpp.py
python3 tests/ci_build/tidy.py --cuda=1  # Include CUDA if applicable
```

**R:**
```bash
cd /path/to/xgboost/
Rscript tests/ci_build/lint_r.R $(pwd)
```

**CMake:**
```bash
bash ./tests/ci_build/lint_cmake.sh
```

#### Run Comprehensive Tests

**Python tests:**
```bash
export PYTHONPATH=./python-package
pytest -v -s --fulltrace tests/python
pytest -v -s --fulltrace tests/python-gpu  # If GPU code modified
pytest -v -s --fulltrace tests/test_distributed  # For federated tests
```

**C++ tests:**
```bash
mkdir build && cd build
cmake -GNinja -DGOOGLE_TEST=ON -DUSE_DMLC_GTEST=ON -DUSE_CUDA=ON -DUSE_NCCL=ON ..
ninja
./testxgboost
```

**Federated-specific tests:**
```bash
# Run federated learning tests
pytest -v -s tests/test_distributed/test_federated/
pytest -v -s tests/test_distributed/test_gpu_federated/
```

### 2. Rebase on Latest Master

Ensure your branch is up-to-date with the upstream master:

```bash
# Add upstream remote (if not already added)
git remote add upstream https://github.com/dmlc/xgboost
git fetch upstream

# Rebase your branch
git checkout fed_secure
git rebase upstream/master

# Resolve any conflicts
# For each conflicted file:
#   - Manually resolve conflicts
#   - git add <resolved_file>
#   - git rebase --continue

# Force push to your fork
git push --force origin fed_secure
```

### 3. Documentation Requirements

Ensure comprehensive documentation:

- [ ] **API Documentation**: All public APIs have docstrings (NumPy format for Python, Doxygen for C++)
- [ ] **Tutorial/Guide**: Add tutorial under `doc/tutorials/` explaining federated learning usage
- [ ] **Parameter Documentation**: Update `doc/parameter.rst` with new federated parameters
- [ ] **Change Log**: Document changes in appropriate release notes
- [ ] **README Updates**: Update main README.md if needed

**Example Documentation Structure:**
```bash
doc/tutorials/federated_learning.rst  # User-facing tutorial
doc/jvm/federated_xgboost4j.rst       # If JVM support added
```

---

## Submission Strategy

Given the large scope of changes (221 files), consider breaking the PR into logical chunks:

### Option 1: Monolithic PR (Recommended for Federated Feature)

Submit as a single comprehensive PR since federated learning is a cohesive feature.

**Advantages:**
- Maintains feature coherence
- Easier to review the complete picture
- CI tests the full integration

**PR Structure:**
- Clear title: `[Feature] Add Secure Federated Learning Support`
- Comprehensive description (see template below)
- Link to any related RFCs or discussions

### Option 2: Staged PRs

If reviewers request, break into:
1. **PR 1:** Core infrastructure (federated plugin interface, communication layer)
2. **PR 2:** Horizontal federated scheme
3. **PR 3:** Vertical federated scheme
4. **PR 4:** GPU acceleration for federated learning
5. **PR 5:** Documentation and examples

---

## Pull Request Template

```markdown
## Description

This PR introduces secure federated learning capabilities to XGBoost, enabling privacy-preserving distributed training across multiple parties without sharing raw data.

### Features Added

- **Federated Plugin Architecture**: Extensible plugin system for federated learning (`plugin/federated/`)
- **Horizontal Federated Learning**: Support for data distributed across rows with secure aggregation
- **Vertical Federated Learning**: Support for features distributed across parties with privacy preservation
- **GPU Acceleration**: CUDA-enabled federated histogram computation and communication
- **Secure Aggregation**: Gradient and histogram aggregation without exposing individual party data

### Key Components

- `plugin/federated/federated_plugin.{cc,h}`: Core plugin interface
- `plugin/federated/federated_hist.{cc,h}`: Federated histogram computation
- `plugin/federated/federated_comm.{cc,cu,h}`: Communication layer
- `plugin/federated/federated_coll.{cc,cu,h}`: Secure collective operations
- `python-package/xgboost/federated.py`: Python API for federated learning
- `tests/test_distributed/test_federated/`: Comprehensive test suite

### Changes Summary

- 221 files changed
- 7,984 insertions, 2,695 deletions
- Commits: 14 (see commit history for details)

### Testing

- [x] All existing tests pass
- [x] New federated learning tests added
- [x] Python tests: `tests/test_distributed/test_federated/`
- [x] GPU tests: `tests/test_distributed/test_gpu_federated/`
- [x] C++ tests: `tests/cpp/plugin/federated/`
- [x] Tested on both CPU and GPU
- [x] Tested horizontal and vertical federated schemes

### Documentation

- [x] API documentation added (docstrings/Doxygen)
- [x] Installation guide updated (`doc/install.rst`)
- [ ] Tutorial added: `doc/tutorials/federated_learning.rst` (TODO if not present)
- [x] Parameters documented
- [x] Example code provided

### Checklist

- [x] Code follows XGBoost C++ style guide (Google C++ style, 100 char lines)
- [x] Python code follows PEP 8 (checked with pylint, black, isort, mypy)
- [x] All linters pass locally
- [x] Rebased on latest master
- [x] No merge conflicts
- [x] Commit messages are clear and descriptive
- [x] Breaking changes documented (if any)

### Performance Impact

- Minimal impact on non-federated workflows
- Federated communication overhead: [provide benchmarks if available]
- GPU acceleration provides [X]% speedup over CPU for federated training

### Related Issues/RFCs

- Closes #XXXX (if applicable)
- Discussed in: [link to discuss.xgboost.ai thread]

### Test Plan

```bash
# Build with federated support
mkdir build && cd build
cmake -DUSE_CUDA=ON -DUSE_NCCL=ON -DPLUGIN_FEDERATED=ON ..
make -j$(nproc)

# Run federated tests
cd ..
export PYTHONPATH=./python-package
pytest -v tests/test_distributed/test_federated/
pytest -v tests/test_distributed/test_gpu_federated/
```

### Breaking Changes

- None (fully backward compatible)

### Migration Guide

N/A - This is a new feature. Existing code continues to work without modification.

---

🤖 Co-Authored-By: [Your contributors]
```

---

## Creating the Pull Request

### Step 1: Push to Your Fork

```bash
# Ensure you're on the fed_secure branch
git checkout fed_secure

# Push to your fork
git push origin fed_secure
```

### Step 2: Create PR via GitHub

1. Navigate to https://github.com/dmlc/xgboost
2. Click "Pull requests" → "New pull request"
3. Click "compare across forks"
4. Set:
   - **base repository:** `dmlc/xgboost`
   - **base branch:** `master`
   - **head repository:** `ZiyueXu77/xgboost`
   - **compare branch:** `fed_secure`
5. Click "Create pull request"
6. Fill in the PR template (see above)
7. Add labels if possible: `feature`, `plugin`, `gpu` (if you have permissions)

### Step 3: Request Reviews

The XGBoost PMC will assign reviewers, but you can proactively request reviews from:
- Committers familiar with plugin architecture
- GPU specialists (if CUDA changes involved)
- Distributed systems experts
- Community members who expressed interest in federated learning

---

## During Code Review

### Best Practices

1. **Respond Promptly**: Address review comments within 48-72 hours
2. **Be Receptive**: Accept constructive criticism gracefully
3. **Explain Decisions**: Provide technical reasoning for design choices
4. **Request Clarification**: If feedback is unclear, ask for specifics
5. **Test Suggested Changes**: Verify any requested modifications work correctly

### Making Changes

```bash
# Make requested changes locally
git checkout fed_secure
# ... edit files ...

# Commit changes
git add <modified_files>
git commit -m "Address review comments: <brief description>

- Fix issue X as suggested by @reviewer
- Refactor Y for better readability
- Add test case for edge case Z
"

# Push updates
git push origin fed_secure
```

### If Combining Commits is Requested

```bash
# Squash last N commits (e.g., 3)
git rebase -i HEAD~3

# In editor: change 'pick' to 'squash' for commits to combine
# Save and edit the combined commit message
# Force push
git push --force origin fed_secure
```

---

## Common Objections and Responses

### "The PR is too large"

**Response:**
"I understand the concern. Federated learning is a cohesive feature where components are tightly coupled. However, I'm happy to split it if that aids review. I propose:
1. Core infrastructure PR
2. Horizontal scheme PR
3. Vertical scheme PR
4. GPU acceleration PR
Would this breakdown work better?"

### "Performance regression detected"

**Response:**
"Thank you for catching this. The regression appears in [specific scenario]. I'll:
1. Profile the code to identify the bottleneck
2. Optimize [specific component]
3. Add benchmarks to prevent future regressions
Expected timeline: [X days]"

### "Documentation is insufficient"

**Response:**
"I'll add:
1. Comprehensive tutorial at `doc/tutorials/federated_learning.rst`
2. Inline code comments explaining the secure aggregation algorithm
3. Example notebook demonstrating real-world usage
Will have this ready by [date]."

### "Tests don't cover edge cases"

**Response:**
"Good catch. I'll add tests for:
1. Network partition scenarios
2. Byzantine party behavior (if applicable)
3. Varying data distributions across parties
4. GPU memory exhaustion handling"

---

## Post-Merge Tasks

After your PR is merged:

1. **Monitor CI**: Watch for any issues in subsequent CI runs
2. **Update Documentation**: Ensure docs website reflects changes
3. **Respond to Issues**: Help users adopting the federated feature
4. **Write Blog Post**: Consider writing about the feature on discuss.xgboost.ai
5. **Follow Up PRs**: Address any technical debt or follow-up items

---

## Continuous Integration (CI)

XGBoost uses multiple CI systems:

### GitHub Actions
- Automatically runs on PR creation/update
- Tests across multiple platforms (Linux, macOS, Windows)
- GPU tests (if modifications affect CUDA code)

### BuildKite
- More extensive testing infrastructure
- Multi-GPU tests: Some tests require manual activation via review comment
  ```
  /gha run mgpu-test
  ```
- Check pipeline status at https://buildkite.com/xgboost

### Expected CI Duration
- **GitHub Actions:** ~30-60 minutes
- **BuildKite:** ~1-2 hours for full pipeline

### If CI Fails

1. **Check Logs:** Click on the failed check to view logs
2. **Reproduce Locally:** Use Docker to reproduce CI environment:
   ```bash
   tests/ci_build/ci_build.sh gpu --use-gpus \
     --build-arg CUDA_VERSION_ARG=11.8 \
     tests/ci_build/build_via_cmake.sh -DUSE_CUDA=ON
   ```
3. **Fix and Push:** Make corrections and push to update PR

---

## Communication Channels

### For Questions/Discussions
- **Forum:** https://discuss.xgboost.ai (preferred for design discussions)
- **Issues:** https://github.com/dmlc/xgboost/issues (for bugs/feature proposals)
- **PR Comments:** For review-specific discussions

### RFC (Request for Comments)
For major changes like federated learning, consider posting an RFC first:
1. Create a GitHub issue with tag `[RFC]`
2. Describe the proposed changes
3. Gather community feedback
4. Reference the RFC in your PR

---

## Timeline Expectations

For a large PR like this:
- **Initial Review:** 1-2 weeks for first committer review
- **Iteration:** 2-4 weeks of back-and-forth (typical for large features)
- **Final Approval:** 1 week after all comments addressed
- **Merge:** Shortly after approval from 1-2 committers

**Total: 4-7 weeks** is reasonable for a feature of this scope.

---

## Key Contacts

While reviews are assigned automatically, these areas have known experts:

- **Plugin Architecture:** Core maintainers
- **Distributed Systems:** Committers with collective algorithm expertise
- **GPU/CUDA:** NVIDIA contributors and GPU specialists
- **Python Bindings:** Python package maintainers

*Check [CONTRIBUTORS.md](https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md) for current committer list.*

---

## Additional Resources

- **XGBoost Contribution Guide:** https://xgboost.readthedocs.io/en/latest/contrib/index.html
- **Coding Style:** https://xgboost.readthedocs.io/en/latest/contrib/coding_guide.html
- **Git Workflow:** https://xgboost.readthedocs.io/en/latest/contrib/git_guide.html
- **Testing Guide:** https://xgboost.readthedocs.io/en/latest/contrib/unit_tests.html
- **Community Guidelines:** https://xgboost.readthedocs.io/en/latest/contrib/community.html

---

## Success Criteria

Your PR is ready to merge when:

- [x] All CI checks pass (green checkmarks)
- [x] At least one committer approval (PMC member preferred)
- [x] All review comments addressed or discussed
- [x] Documentation complete
- [x] No outstanding objections from reviewers
- [x] No merge conflicts with master

---

## Notes

- **Be Patient:** Large PRs take time to review thoroughly
- **Stay Engaged:** Respond to comments to keep the review momentum
- **Help Others:** Review other PRs to build community goodwill
- **Trust the Process:** XGBoost maintainers are committed to quality and will work with you

**Good luck with your upstream contribution! The XGBoost community appreciates your work on federated learning.** 🚀
