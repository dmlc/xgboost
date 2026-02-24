# Quick Checklist: Upstreaming fed_secure Branch

Use this checklist alongside [UPSTREAM_GUIDE.md](./UPSTREAM_GUIDE.md) for a quick reference.

## Pre-Submission (Do First)

### Code Quality
- [ ] Run Python linter: `python ./tests/ci_build/lint_python.py --fix`
- [ ] Run C++ linter: `python ./tests/ci_build/lint_cpp.py`
- [ ] Run clang-tidy: `python3 tests/ci_build/tidy.py --cuda=1`
- [ ] Run R linter: `Rscript tests/ci_build/lint_r.R $(pwd)`
- [ ] Run CMake linter: `bash ./tests/ci_build/lint_cmake.sh`

### Testing
- [ ] Run Python tests: `pytest -v tests/python`
- [ ] Run GPU tests: `pytest -v tests/python-gpu`
- [ ] Run federated tests: `pytest -v tests/test_distributed/test_federated/`
- [ ] Run GPU federated tests: `pytest -v tests/test_distributed/test_gpu_federated/`
- [ ] Run C++ tests: `./testxgboost` (in build directory)
- [ ] All tests pass locally

### Git Workflow
- [ ] Add upstream remote: `git remote add upstream https://github.com/dmlc/xgboost`
- [ ] Fetch latest: `git fetch upstream`
- [ ] Rebase on master: `git rebase upstream/master`
- [ ] Resolve all conflicts
- [ ] Force push to fork: `git push --force origin fed_secure`
- [ ] No merge conflicts remain

### Documentation
- [ ] All public APIs have docstrings/Doxygen comments
- [ ] Parameters documented in `doc/parameter.rst`
- [ ] Tutorial added or planned for `doc/tutorials/`
- [ ] README.md updated if needed
- [ ] Change log entry drafted

## Pull Request Creation

### PR Description
- [ ] Clear title: `[Feature] Add Secure Federated Learning Support`
- [ ] Comprehensive description with:
  - [ ] Feature summary
  - [ ] Key components list
  - [ ] Testing checklist
  - [ ] Documentation checklist
  - [ ] Performance impact notes
  - [ ] Breaking changes (if any)

### PR Setup
- [ ] Base: `dmlc/xgboost:master`
- [ ] Head: `ZiyueXu77/xgboost:fed_secure`
- [ ] Labels added (if possible): `feature`, `plugin`, `gpu`
- [ ] Reviewers requested (if applicable)

## During Review

### Response
- [ ] Respond to comments within 48-72 hours
- [ ] Test all suggested changes before committing
- [ ] Push updates after addressing feedback
- [ ] Request clarification on unclear comments

### Commits
- [ ] Commit messages are clear and descriptive
- [ ] Reference review comments in commit messages
- [ ] Squash commits if requested by reviewers

## CI Monitoring

### Check Status
- [ ] GitHub Actions: all checks pass ✅
- [ ] BuildKite: pipeline succeeds ✅
- [ ] No linter failures
- [ ] No test failures

### If CI Fails
- [ ] Review failure logs
- [ ] Reproduce locally if possible
- [ ] Fix issues and push updates

## Before Merge

### Final Checks
- [ ] All CI checks green
- [ ] At least 1 committer approval
- [ ] All review comments resolved
- [ ] Documentation complete
- [ ] No merge conflicts
- [ ] No outstanding reviewer objections

## Post-Merge

### Follow-Up
- [ ] Monitor CI on master branch
- [ ] Watch for related issues
- [ ] Respond to user questions
- [ ] Address any follow-up items
- [ ] Consider blog post or tutorial

---

## Quick Commands Reference

### Setup
```bash
git remote add upstream https://github.com/dmlc/xgboost
git fetch upstream
```

### Rebase
```bash
git checkout fed_secure
git rebase upstream/master
# Resolve conflicts, then:
git push --force origin fed_secure
```

### Run All Linters
```bash
python ./tests/ci_build/lint_python.py --fix
python ./tests/ci_build/lint_cpp.py
python3 tests/ci_build/tidy.py --cuda=1
Rscript tests/ci_build/lint_r.R $(pwd)
bash ./tests/ci_build/lint_cmake.sh
```

### Run All Tests
```bash
export PYTHONPATH=./python-package
pytest -v tests/python
pytest -v tests/python-gpu
pytest -v tests/test_distributed/test_federated/
pytest -v tests/test_distributed/test_gpu_federated/
cd build && ./testxgboost
```

### Squash Commits
```bash
git rebase -i HEAD~N  # N = number of commits
# Change 'pick' to 'squash' in editor
git push --force origin fed_secure
```

---

## Estimated Timeline

| Phase | Duration |
|-------|----------|
| Pre-submission prep | 1-3 days |
| Initial review | 1-2 weeks |
| Review iterations | 2-4 weeks |
| Final approval | 1 week |
| **Total** | **4-7 weeks** |

---

## Help & Resources

- 📖 Full Guide: [UPSTREAM_GUIDE.md](./UPSTREAM_GUIDE.md)
- 🌐 Contribution Docs: https://xgboost.readthedocs.io/en/latest/contrib/
- 💬 Forum: https://discuss.xgboost.ai
- 🐛 Issues: https://github.com/dmlc/xgboost/issues

---

**Status Tracking**

Current Status: ⬜ Not Started / 🟡 In Progress / ✅ Complete

- ⬜ Pre-submission checks
- ⬜ PR created
- ⬜ First review received
- ⬜ All feedback addressed
- ⬜ CI passing
- ⬜ Approved
- ⬜ Merged

---

*Last updated: [Date]*
