#!/bin/bash
set -e
set -x

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
    python setup.py install --user
    popd
  fi
}

function uninstall_xgboost {
  pip uninstall -y xgboost
}

# Run specified test suite
case "$suite" in
  gpu)
    source activate gpu_test
    install_xgboost
    pytest -v -s -rxXs --fulltrace --durations=0 -m "not mgpu" ${args} tests/python-gpu
    uninstall_xgboost
    ;;

  mgpu)
    source activate gpu_test
    install_xgboost
    pytest -v -s -rxXs --fulltrace --durations=0 -m "mgpu" ${args} tests/python-gpu

    cd tests/distributed
    ./runtests-gpu.sh
    uninstall_xgboost
    ;;

  cpu)
    source activate cpu_test
    install_xgboost
    pytest -v -s -rxXs --fulltrace --durations=0 ${args} tests/python
    cd tests/distributed
    ./runtests.sh
    uninstall_xgboost
    ;;

  cpu-arm64)
    source activate aarch64_test
    install_xgboost
    pytest -v -s -rxXs --fulltrace --durations=0 ${args} tests/python/test_basic.py tests/python/test_basic_models.py tests/python/test_model_compatibility.py
    uninstall_xgboost
    ;;

  *)
    echo "Usage: $0 {gpu|mgpu|cpu|cpu-arm64} [extra args to pass to pytest]"
    exit 1
    ;;
esac
