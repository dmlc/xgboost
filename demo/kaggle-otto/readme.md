Benckmark for Otto Group Competition
=========

This is a folder containing the benchmark for the [Otto Group Competition on Kaggle](http://www.kaggle.com/c/otto-group-product-classification-challenge).

## Getting started

1. Put `train.csv` and `test.csv` under the `data` folder
2. Run the script
3. Submit the `submission.csv`

The parameter `nthread` controls the number of cores to run on, please set it to suit your machine.

## R-package

To install the R-package of xgboost, please run

```r
devtools::install_github('tqchen/xgboost',subdir='R-package')
```

Windows users may need to install [RTools](http://cran.r-project.org/bin/windows/Rtools/) first.


