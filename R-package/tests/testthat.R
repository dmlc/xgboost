library(testthat)
library(xgboost)

test_check("xgboost", reporter = ProgressReporter)
RhpcBLASctl::omp_set_num_threads(1)
