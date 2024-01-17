This folder contains test cases for XGBoost c++ core, Python package and some other CI
facilities.

# Directories
  * ci_build:  Test facilities for Jenkins CI and GitHub action.
  * cli: Basic test for command line executable `xgboost`.  Most of the other command line
    specific tests are in Python test `test_cli.py`.
  * cpp: Tests for C++ core, using Google test framework.
  * python: Tests for Python package, demonstrations and CLI.  For how to setup the
    dependencies for tests, see conda files in `ci_build`.
  * python-gpu: Similar to python tests, but for GPU.
  * travis: CI facilities for Travis.
  * test_distributed: Test for distributed systems including spark and dask.

# Others
  * pytest.ini: Describes the `pytest` marker for python tests, some markers are generated
    by `conftest.py` file.
