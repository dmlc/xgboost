#!/bin/bash

if [[ $# -ne 1 ]]
then
  echo "Usage: $0 [branch name]"
  exit 1
fi

branch_name=$1

if [[ -z "${R_LIBS_USER}" ]];
then
    export R_LIBS_USER=/tmp/rtmpdir
fi

set -euo pipefail

echo "R_LIBS_USER: ${R_LIBS_USER}"

if [[ ! -d ${R_LIBS_USER} ]]
then
    echo "Make ${R_LIBS_USER} for installing temporary R packages."
    mkdir ${R_LIBS_USER}
fi

# Used only in container environment
if command -v gosu 2>&1 >/dev/null
then
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

cd ../../			# back to project root

tar cvjf r-docs-${branch_name}.tar.bz2 R-package/docs doc/R-package/xgboost_introduction.md doc/R-package/xgboostfromJSON.md
