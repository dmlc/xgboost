XGBoost R Package for Scalable GBM
==================================

[![CRAN Status Badge](http://www.r-pkg.org/badges/version/xgboost)](https://cran.r-project.org/web/packages/xgboost)
[![CRAN Downloads](http://cranlogs.r-pkg.org/badges/xgboost)](https://cran.rstudio.com/web/packages/xgboost/index.html)
[![Documentation Status](https://readthedocs.org/projects/xgboost/badge/?version=latest)](http://xgboost.readthedocs.org/en/latest/R-package/index.html)

Resources
---------
* [XGBoost R Package Online Documentation](http://xgboost.readthedocs.org/en/latest/R-package/index.html)
  - Check this out for detailed documents, examples and tutorials.

Installation
------------

We are [on CRAN](https://cran.r-project.org/web/packages/xgboost/index.html) now. For stable/pre-compiled(for Windows and OS X) version, please install from CRAN:

```r
install.packages('xgboost')
```

You can also install from our weekly updated drat repo:
```r
install.packages("xgboost", repos=c("http://dmlc.ml/drat/", getOption("repos")), type="source")
```

***Important*** Due to the usage of submodule, `install_github` is no longer support to install the
latest version of R package. 
For up-to-date version, please install from github.

Windows users will need to install [RTools](https://cran.r-project.org/bin/windows/Rtools/) first. They also need to download [MinGW-W64](http://iweb.dl.sourceforge.net/project/mingw-w64/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/installer/mingw-w64-install.exe) using x86_64 architecture during installation.

Run the following command to add MinGW to PATH in Windows if not already added.

```cmd
PATH %PATH%;C:\Program Files\mingw-w64\x86_64-5.3.0-posix-seh-rt_v4-rev0\mingw64\bin
```

To compile xgboost at the root of your storage, run the following bash script.

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
git submodule init
git submodule update
alias make='mingw32-make'
cd dmlc-core
make -j4
cd ../rabit
make lib/librabit_empty.a -j4
cd ..
cp make/mingw64.mk config.mk
make -j4
```

Run the following R script to install xgboost package from the root directory.

```r
install.package('devtools') # if not installed
setwd('C:/xgboost/')
library(devtools)
install('R-package')
```

For more detailed installation instructions, please see [here](http://xgboost.readthedocs.org/en/latest/build.html#r-package-installation).

Examples
--------

* Please visit [walk through example](demo).
* See also the [example scripts](../demo/kaggle-higgs) for Kaggle Higgs Challenge, including [speedtest script](../demo/kaggle-higgs/speedtest.R) on this dataset and the one related to [Otto challenge](../demo/kaggle-otto), including a [RMarkdown documentation](../demo/kaggle-otto/understandingXGBoostModel.Rmd).
