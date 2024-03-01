library(testthat)
library(xgboost)
library(Matrix)

test_check("xgboost", reporter = ProgressReporter)
RhpcBLASctl::omp_set_num_threads(1)
