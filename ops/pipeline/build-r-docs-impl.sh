#!/bin/bash

if [[ $# -ne 1 ]]
then
  echo "Usage: $0 [branch name]"
  exit 1
fi

set -euo pipefail

branch_name=$1

echo "R_LIBS_USER: ${R_LIBS_USER}"

gosu root chown -R $UID:$GROUPS ${R_LIBS_USER}

cd R-package
MAKEFLAGS=-j$(nproc) Rscript ./tests/helper_scripts/install_deps.R
# Some examples are failing
MAKEFLAGS=-j$(nproc) Rscript -e "pkgdown::build_site(examples=FALSE)"
cd -

cd doc/R-package
make -j$(nproc) all

cd ../../			# back to project root

tar cvjf r-docs-${branch_name}.tar.bz2 R-package/docs doc/R-package/xgboost_introduction.md doc/R-package/xgboostfromJSON.md