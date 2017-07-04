#!/usr/bin/env bash

mkdir build
cd build
cmake .. -DPLUGIN_UPDATER_GPU=ON
make
cd ..
cd python-package
python setup.py install --user
cd ../plugin/updater_gpu
python -m nose test/python
