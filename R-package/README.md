R package for xgboost
=====================

Installation
------------

For up-to-date version (which is recommended), please install from github. Windows user will need to install [RTools](http://cran.r-project.org/bin/windows/Rtools/) first.

```r
devtools::install_github('dmlc/xgboost',subdir='R-package')
```

Examples
--------

* Please visit [walk through example](demo).
* See also the [example scripts](../demo/kaggle-higgs) for Kaggle Higgs Challenge, including [speedtest script](../demo/kaggle-higgs/speedtest.R) on this dataset and the one related to [Otto challenge](../demo/kaggle-otto), including a [RMarkdown documentation](../demo/kaggle-otto/understandingXGBoostModel.Rmd).

Notes
-----

If you face an issue installing the package using  ```devtools::install_github```, something like this (even after updating libxml and RCurl as lot of forums say) -

```
devtools::install_github('dmlc/xgboost',subdir='R-package')
Downloading github repo dmlc/xgboost@master
Error in function (type, msg, asError = TRUE)  : 
  Peer certificate cannot be authenticated with given CA certificates
```
To get around this you can build the package locally as mentioned [here](https://github.com/dmlc/xgboost/issues/347) - 
```
1. Clone the current repository and set your workspace to xgboost/R-package/
2. Run R CMD INSTALL --build . in terminal to get the tarball.
3. Run install.packages('path_to_the_tarball',repo=NULL) in R to install.
```
