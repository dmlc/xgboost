#!/bin/bash

# To be called when R package tests have failed

set -e
set -x

if [ -f "xgboost.Rcheck/00install.out" ]; then
  echo "===== xgboost.Rcheck/00install.out ===="
  cat xgboost.Rcheck/00install.out
fi

if [ -f "xgboost.Rcheck/00install.log" ]; then
  echo "\n\n===== xgboost.Rcheck/00install.log ===="
  cat xgboost.Rcheck/00install.log
fi

# Produce error code to interrupt Jenkins pipeline
exit 1
