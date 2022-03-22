#!/bin/bash

# To be called when R package tests have failed

set -e
set -x

flag="$1"

if [ -f "xgboost.Rcheck/00install.out" ]; then
  echo "===== xgboost.Rcheck/00install.out ===="
  cat xgboost.Rcheck/00install.out
fi

if [ -f "xgboost.Rcheck/00check.log" ]; then
  printf "\n\n===== xgboost.Rcheck/00check.log ====\n"
  cat xgboost.Rcheck/00check.log
fi

if [[ "$flag" == "fail" ]]
then
  exit 1
fi
