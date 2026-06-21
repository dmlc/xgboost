library(testthat)
library(xgboost)
library(Matrix)

RhpcBLASctl::omp_set_num_threads(1)
data.table::setDTthreads(1)
test_check("xgboost", reporter = ProgressReporter)
