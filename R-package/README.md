# R package for xgboost.

## Installation

For up-to-date version(which is recommended), please install from github. Windows user will need to install [RTools](http://cran.r-project.org/bin/windows/Rtools/) first.

```r
require(devtools)
install_github('xgboost','tqchen',subdir='R-package')
```

For stable version on CRAN, please run

```r
install.packages('xgboost')
```

## Examples

* Please visit [walk through example](https://github.com/tqchen/xgboost/blob/master/R-package/demo).
* See also the [example scripts](https://github.com/tqchen/xgboost/tree/master/demo/kaggle-higgs) for Kaggle Higgs Challenge, including [speedtest script](https://github.com/tqchen/xgboost/blob/master/demo/kaggle-higgs/speedtest.R) on this dataset.
