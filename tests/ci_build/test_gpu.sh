#!/usr/bin/env bash

cd python-package
python setup.py install --user
cd ..
python -m nose tests/python-gpu

