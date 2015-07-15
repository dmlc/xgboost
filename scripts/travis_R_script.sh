#!/bin/bash
# Test R package of xgboost
set -e
export _R_CHECK_TIMINGS_=0
export R_BUILD_ARGS="--no-build-vignettes --no-manual"
export R_CHECK_ARGS="--no-vignettes --no-manual"

curl -OL http://raw.github.com/craigcitro/r-travis/master/scripts/travis-tool.sh
chmod 755 ./travis-tool.sh
./travis-tool.sh bootstrap
make Rpack
cd ./xgboost
../travis-tool.sh install_deps
../travis-tool.sh run_tests