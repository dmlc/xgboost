#!/bin/bash
set -e
set -x

suite=$1

# Install XGBoost Python package
wheel_found=0
for file in python-package/dist/*.whl
do
  pip install --user ${file}
  wheel_found=1
  break  # need just one
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

  cpu)
    pytest -v -s --fulltrace tests/python
    ;;

  *)
    echo "Usage: $0 {gpu|mgpu|cpu}"
    exit 1
    ;;
esac
