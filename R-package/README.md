# R package for xgboost.

## Installation

For up-to-date version (which is recommended), please install from github. Windows user will need to install [RTools](http://cran.r-project.org/bin/windows/Rtools/) first.

```r
devtools::install_github('dmlc/xgboost',subdir='R-package')
```

For stable version on CRAN, please run

```r
install.packages('xgboost')
```

## Examples

* Please visit [walk through example](demo).
* See also the [example scripts](../demo/kaggle-higgs) for Kaggle Higgs Challenge, including [speedtest script](../demo/kaggle-higgs/speedtest.R) on this dataset and the one related to [Otto challenge](../demo/kaggle-otto), including a [RMarkdown documentation](../demo/kaggle-otto/understandingXGBoostModel.Rmd).
