#!/bin/bash

if [[ $# -ne 1 ]]
then
  echo "Usage: $0 [branch name]"
  exit 1
fi

set -euo pipefail

branch_name=$1

# See instructions at: https://cran.r-project.org/bin/linux/ubuntu/

wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | sudo tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
# add the R 4.0 repo from CRAN -- adjust 'focal' to 'groovy' or 'bionic' as needed
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

sudo apt install --no-install-recommends r-base
Rscript -e "install.packages(c('pkgdown'), repos = 'https://mirror.las.iastate.edu/CRAN/')"
cd R-package
Rscript -e "pkgdown::build_site()"
cd -
tar cvjf r-docs-${branch_name}.tar.bz2 R-package/docs
