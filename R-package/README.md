XGBoost R Package for Scalable GBM
==================================

[![CRAN Status Badge](http://www.r-pkg.org/badges/version/xgboost)](http://cran.r-project.org/web/packages/xgboost)
[![CRAN Downloads](http://cranlogs.r-pkg.org/badges/xgboost)](http://cran.rstudio.com/web/packages/xgboost/index.html)
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
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type="source")
```

***Important*** Due to the usage of submodule, `install_github` is no longer support to install the
latest version of R package. 
For up-to-date version, please install from github. Windows user will need to install [RTools](http://cran.r-project.org/bin/windows/Rtools/) first.

```bash
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost
make Rbuild
R CMD INSTALL xgboost_0.4-3.tar.gz
```

For more detailed installation instructions, please see [here](http://xgboost.readthedocs.org/en/latest/build.html#r-package-installation).

Examples
--------

* Please visit [walk through example](demo).
* See also the [example scripts](../demo/kaggle-higgs) for Kaggle Higgs Challenge, including [speedtest script](../demo/kaggle-higgs/speedtest.R) on this dataset and the one related to [Otto challenge](../demo/kaggle-otto), including a [RMarkdown documentation](../demo/kaggle-otto/understandingXGBoostModel.Rmd).
