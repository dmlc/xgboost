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

  mgpu)
    source activate gpu_test
    install_xgboost
    pytest -v -s -rxXs --fulltrace -m "mgpu" tests/python-gpu

    cd tests/distributed
    ./runtests-gpu.sh
    cd -
    ;;

  cudf)
    source activate cudf_test
    install_xgboost
    pytest -v -s -rxXs --fulltrace -m "not mgpu" \
           tests/python-gpu/test_from_cudf.py tests/python-gpu/test_from_cupy.py \
	   tests/python-gpu/test_gpu_prediction.py
    ;;

  mgpu-cudf)
    source activate cudf_test
    install_xgboost
    pytest -v -s -rxXs --fulltrace -m "mgpu" tests/python-gpu/test_gpu_with_dask.py
    ;;

  cpu)
    install_xgboost
    pytest -v -s --fulltrace tests/python
    cd tests/distributed
    ./runtests.sh
    ;;

  *)
    echo "Usage: $0 {gpu|mgpu|cudf|cpu}"
    exit 1
    ;;
esac
