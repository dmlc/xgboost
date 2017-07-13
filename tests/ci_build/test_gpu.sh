#!/usr/bin/env bash

cd python-package
python setup.py install --user
cd ../plugin/updater_gpu
python -m nose test/python
