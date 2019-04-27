#!/bin/bash

set -e
set -x

make Rpack
cd xgboost/

# Run tests
echo "Building with R CMD build"
R CMD build --no-build-vignettes --no-manual .

echo "Running R tests"
R_PACKAGE_TARBALL=$(ls -1t *.tar.gz | head -n 1)

export _R_CHECK_TIMINGS_=0
export _R_CHECK_FORCE_SUGGESTS_=false
R CMD check \
  ${R_PACKAGE_TARBALL} \
  --no-vignettes \
  --no-manual \
  --as-cran \
  --install-args=--build
