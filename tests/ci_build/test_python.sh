#!/bin/bash
set -e
set -x

suite=$1

# Install XGBoost Python package
wheel_found=0
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

# Run specified test suite
case "$suite" in
  gpu)
    pytest -v -s --fulltrace -m "(not slow) and (not mgpu)" tests/python-gpu
    ;;

  mgpu)
    pytest -v -s --fulltrace -m "(not slow) and mgpu" tests/python-gpu
    cd tests/distributed
    ./runtests-gpu.sh
    ;;

  cudf)
    source activate cudf_test
    python -m pytest -v -s --fulltrace tests/python-gpu/test_from_columnar.py tests/python-gpu/test_gpu_with_dask.py
    ;;

  cpu)
    pytest -v -s --fulltrace tests/python
    cd tests/distributed
    ./runtests.sh
    ;;

  *)
    echo "Usage: $0 {gpu|mgpu|cudf|cpu}"
    exit 1
    ;;
esac
