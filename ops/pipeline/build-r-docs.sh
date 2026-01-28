#!/bin/bash
## Build docs for the R package and package it in a tarball.

set -euo pipefail

if [[ -z ${BRANCH_NAME:-} ]]; then
  echo "Make sure to define environment variable BRANCH_NAME."
  exit 1
fi

if [[ -z "${R_LIBS_USER:-}" ]]; then
  export R_LIBS_USER=/tmp/rtmpdir
fi

echo "--- Build R package doc"
echo "R_LIBS_USER: ${R_LIBS_USER}"
set -x

if [[ ! -d ${R_LIBS_USER} ]]; then
  echo "Make ${R_LIBS_USER} for installing temporary R packages."
  mkdir ${R_LIBS_USER}
fi

# Used only in container environment
if command -v gosu 2>&1 >/dev/null; then
  gosu root chown -R $UID:$GROUPS ${R_LIBS_USER}
fi

cd R-package

MAKEFLAGS=-j$(nproc) Rscript ./tests/helper_scripts/install_deps.R
# Some examples are failing
MAKEFLAGS=-j$(nproc) Rscript -e "pkgdown::build_site(examples=FALSE)"
# Install the package for vignettes
MAKEFLAGS=-j$(nproc) R CMD INSTALL .

cd -

cd doc/R-package
make -j$(nproc) all

cd ../../  # back to project root

tar cvjf r-docs-${BRANCH_NAME}.tar.bz2 R-package/docs doc/R-package/xgboost_introduction.md doc/R-package/xgboostfromJSON.md
