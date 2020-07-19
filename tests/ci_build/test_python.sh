#!/bin/bash
set -e
set -x

suite=$1

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

# Run specified test suite
case "$suite" in
  gpu)
    source activate gpu_test
    install_xgboost
    pytest -v -s -rxXs --fulltrace -m "not mgpu" tests/python-gpu
    ;;

  gpu-rmm)
    source activate gpu_test
    install_xgboost
    pytest -v -s -rxXs tests/python-gpu/test_gpu_demos.py::test_rmm_integration_demo
    XGBOOST_RMM_TEST_LIBPATH=demo/rmm-integration/build/librmm_bridge.so \
      pytest -v -s -rxXs --fulltrace -m "not mgpu" -k "not test_rmm_integration_demo" \
      tests/python-gpu/ --ignore=tests/python-gpu/test_gpu_pickling.py
    ;;

  mgpu)
    source activate gpu_test
    install_xgboost
    pytest -v -s -rxXs --fulltrace -m "mgpu" tests/python-gpu

    cd tests/distributed
    ./runtests-gpu.sh
    cd -
    ;;

  cpu)
    install_xgboost
    pytest -v -s --fulltrace tests/python
    cd tests/distributed
    ./runtests.sh
    ;;

  *)
    echo "Usage: $0 {gpu|gpu-rmm|mgpu|cpu}"
    exit 1
    ;;
esac
