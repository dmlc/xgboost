require(xgboost)
require(Matrix)

context("testing xgb.DMatrix functionality")

data(agaricus.test, package='xgboost')
test_data <- agaricus.test$data[1:100,]
test_label <- agaricus.test$label[1:100]

test_that("xgb.DMatrix: basic construction", {
  # from sparse matrix
  dtest1 <- xgb.DMatrix(test_data, label=test_label)
  
  # from dense matrix 
  dtest2 <- xgb.DMatrix(as.matrix(test_data), label=test_label)
  expect_equal(getinfo(dtest1, 'label'), getinfo(dtest2, 'label'))
  expect_equal(dim(dtest1), dim(dtest2))
  
  #from dense integer matrix
  int_data <- as.matrix(test_data)
  storage.mode(int_data) <- "integer"
  dtest3 <- xgb.DMatrix(int_data, label=test_label)
  expect_equal(dim(dtest1), dim(dtest3))
})

test_that("xgb.DMatrix: saving, loading", {
  # save to a local file
  dtest1 <- xgb.DMatrix(test_data, label=test_label)
  tmp_file <- tempfile('xgb.DMatrix_')
  expect_true(xgb.DMatrix.save(dtest1, tmp_file))
  # read from a local file
  expect_output(dtest3 <- xgb.DMatrix(tmp_file), "entries loaded from")
  expect_output(dtest3 <- xgb.DMatrix(tmp_file, silent = TRUE), NA)
  unlink(tmp_file)
  expect_equal(getinfo(dtest1, 'label'), getinfo(dtest3, 'label'))
  
  # from a libsvm text file
  tmp <- c("0 1:1 2:1","1 3:1","0 1:1")
  tmp_file <- 'tmp.libsvm'
  writeLines(tmp, tmp_file)
  dtest4 <- xgb.DMatrix(tmp_file, silent = TRUE)
  expect_equal(dim(dtest4), c(3, 4))
  expect_equal(getinfo(dtest4, 'label'), c(0,1,0))
  unlink(tmp_file)
})

test_that("xgb.DMatrix: getinfo & setinfo", {
  dtest <- xgb.DMatrix(test_data)
  expect_true(setinfo(dtest, 'label', test_label))
  labels <- getinfo(dtest, 'label')
  expect_equal(test_label, getinfo(dtest, 'label'))
  
  expect_true(length(getinfo(dtest, 'weight')) == 0)
  expect_true(length(getinfo(dtest, 'base_margin')) == 0)

  expect_true(setinfo(dtest, 'weight', test_label))
  expect_true(setinfo(dtest, 'base_margin', test_label))
  expect_true(setinfo(dtest, 'group', c(50,50)))
  expect_error(setinfo(dtest, 'group', test_label))
  
  # providing character values will give a warning
  expect_warning( setinfo(dtest, 'weight', rep('a', nrow(test_data))) )
  
  # any other label should error
  expect_error(setinfo(dtest, 'asdf', test_label))
})

test_that("xgb.DMatrix: slice, dim", {
  dtest <- xgb.DMatrix(test_data, label=test_label)
  expect_equal(dim(dtest), dim(test_data))
  dsub1 <- slice(dtest, 1:42)
  expect_equal(nrow(dsub1), 42)
  expect_equal(ncol(dsub1), ncol(test_data))
  
  dsub2 <- dtest[1:42,]
  expect_equal(dim(dtest), dim(test_data))
  expect_equal(getinfo(dsub1, 'label'), getinfo(dsub2, 'label'))
})

test_that("xgb.DMatrix: colnames", {
  dtest <- xgb.DMatrix(test_data, label=test_label)
  expect_equal(colnames(dtest), colnames(test_data))
  expect_error( colnames(dtest) <- 'asdf')
  new_names <- make.names(1:ncol(test_data))
  expect_silent( colnames(dtest) <- new_names)
  expect_equal(colnames(dtest), new_names)
  expect_silent(colnames(dtest) <- NULL)
  expect_null(colnames(dtest))
})

test_that("xgb.DMatrix: nrow is correct for a very sparse matrix", {
  set.seed(123)
  nr <- 1000
  x <- rsparsematrix(nr, 100, density=0.0005)
  # we want it very sparse, so that last rows are empty
  expect_lt(max(x@i), nr)
  dtest <- xgb.DMatrix(x)
  expect_equal(dim(dtest), dim(x))
})
