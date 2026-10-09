#!/bin/bash
# Intel provides wheel/runtime compatibility coverage; ARM runs the full suite.
# Keep this allowlist focused to target a 10–12 minute Intel job including setup.
# Revisit it when Intel runner performance improves or platform-specific regressions
# require more coverage. New modules join the ARM suite automatically.
set -euo pipefail

case "${1:-}" in
  full)
    python -m pytest -s -v -rxXs --durations=0 tests/python
    python -m pytest -s -v -rxXs --durations=0 tests/test_distributed/test_with_dask
    ;;
  basic)
    # Whole modules cover native loading/OpenMP, core algorithms, model I/O,
    # and Python data interfaces. Heavy scenario matrices have smoke tests below.
    python -m pytest -s -v -rxXs --durations=0 \
      tests/python/test_basic.py \
      tests/python/test_basic_models.py \
      tests/python/test_callback.py \
      tests/python/test_collective.py \
      tests/python/test_config.py \
      tests/python/test_dmatrix.py \
      tests/python/test_early_stopping.py \
      tests/python/test_eval_metrics.py \
      tests/python/test_intercept.py \
      tests/python/test_interaction_constraints.py \
      tests/python/test_interpret.py \
      tests/python/test_linear.py \
      tests/python/test_model_compatibility.py \
      tests/python/test_model_io.py \
      tests/python/test_monotone_constraints.py \
      tests/python/test_openmp.py \
      tests/python/test_parse_tree.py \
      tests/python/test_pickling.py \
      tests/python/test_plotting.py \
      tests/python/test_predict.py \
      tests/python/test_quantile_dmatrix.py \
      tests/python/test_ranking.py \
      tests/python/test_shap.py \
      tests/python/test_survival.py \
      tests/python/test_training_continuation.py \
      tests/python/test_updaters.py \
      tests/python/test_with_arrow.py \
      tests/python/test_with_pandas.py \
      tests/python/test_with_scipy.py \
      tests/python/test_with_sklearn.py \
      tests/python/test_tracker.py::test_rabit_tracker \
      tests/python/test_ordinal.py::test_cat_container \
      tests/python/test_ordinal.py::test_cat_thread_safety \
      tests/python/test_multi_target.py::test_multiclass \
      tests/python/test_multi_target.py::test_multilabel \
      tests/python/test_data_iterator.py::test_single_batch \
      tests/python/test_data_iterator.py::test_categorical_extmem_qdm
    # Exercise actual distributed training with array and dataframe input, without
    # repeating the exhaustive Dask and process-pool matrices on Intel.
    python -m pytest -s -v -rxXs --durations=0 \
      tests/test_distributed/test_with_dask/test_with_dask.py::test_from_dask_array \
      tests/test_distributed/test_with_dask/test_with_dask.py::test_from_dask_dataframe
    ;;
  *)
    echo "Usage: $0 {basic|full}" >&2
    exit 2
    ;;
esac
