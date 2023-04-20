#!/bin/bash
set -e

if [ "$#" -lt 1 ]
then
  suite=''
  args=''
else
  suite=$1
  shift 1
  args="$@"
fi

# Install XGBoost Python package
function install_xgboost {
  wheel_found=0
  pip install --upgrade pip --user
  for file in python-package/dist/*.whl
  do
    if [ -e "${file}" ]
    then
      pip install --user "${file}"
      wheel_found=1
      break  # need just one
    fi
  done
  if [ "$wheel_found" -eq 0 ]
  then
    pushd .
    cd python-package
    pip install --user -v .
    popd
  fi
}

function setup_pyspark_envs {
  export PYSPARK_DRIVER_PYTHON=`which python`
  export PYSPARK_PYTHON=`which python`
  export SPARK_TESTING=1
}

function unset_pyspark_envs {
  unset PYSPARK_DRIVER_PYTHON
  unset PYSPARK_PYTHON
  unset SPARK_TESTING
}

function uninstall_xgboost {
  pip uninstall -y xgboost
}

# Run specified test suite
case "$suite" in
  gpu)
    source activate gpu_test
    set -x
    install_xgboost
    setup_pyspark_envs
    pytest -v -s -rxXs --fulltrace --durations=0 -m "not mgpu" ${args} tests/python-gpu
    unset_pyspark_envs
    uninstall_xgboost
    set +x
    ;;

  mgpu)
    source activate gpu_test
    set -x
    install_xgboost
    setup_pyspark_envs
    pytest -v -s -rxXs --fulltrace --durations=0 -m "mgpu" ${args} tests/python-gpu
    pytest -v -s -rxXs --fulltrace --durations=0 -m "mgpu" ${args} tests/test_distributed/test_gpu_with_dask
    pytest -v -s -rxXs --fulltrace --durations=0 -m "mgpu" ${args} tests/test_distributed/test_gpu_with_spark
    unset_pyspark_envs
    uninstall_xgboost
    set +x
    ;;

  cpu)
    source activate linux_cpu_test
    set -x
    install_xgboost
    export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
    setup_pyspark_envs
    pytest -v -s -rxXs --fulltrace --durations=0 ${args} tests/python
    pytest -v -s -rxXs --fulltrace --durations=0 ${args} tests/test_distributed/test_with_dask
    pytest -v -s -rxXs --fulltrace --durations=0 ${args} tests/test_distributed/test_with_spark
    unset_pyspark_envs
    uninstall_xgboost
    set +x
    ;;

  cpu-arm64)
    source activate aarch64_test
    set -x
    install_xgboost
    setup_pyspark_envs
    pytest -v -s -rxXs --fulltrace --durations=0 ${args} tests/python/test_basic.py tests/python/test_basic_models.py tests/python/test_model_compatibility.py
    unset_pyspark_envs
    uninstall_xgboost
    set +x
    ;;

  *)
    echo "Usage: $0 {gpu|mgpu|cpu|cpu-arm64} [extra args to pass to pytest]"
    exit 1
    ;;
esac
