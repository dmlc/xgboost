library(testthat)
library(xgboost)

test_check("xgboost", reporter = ProgressReporter)
