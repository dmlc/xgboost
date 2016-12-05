#!/bin/bash

PYTHONPATH=../../python-package/ ../../dmlc-core/tracker/dmlc-submit  --cluster=local --num-workers=3\
  python test_basic.py
