context("dim and automatic feature names")

library("methods")
library("xgboost")

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')

train <- agaricus.train
test <- agaricus.test

dtrain <- xgb.DMatrix(data = train$data, label = train$label)

bst <- xgboost(data = dtrain, 
               max.depth = 2, 
               eta = 1, 
               nround = 10, 
               nthread = 1, 
               verbose = 0,
               objective = "binary:logistic")

test_that("dim is correct", {
  expect_equal(dim(dtrain), c(6513, 126))
})

test_that("feature names automatically populated for importance", {
  expect_equal(xgb.importance(model = bst)$Feature[1], "odor=none")
})

attr(dtrain, "dimensions") <- NULL
test_that("dim doesn't fail if attributes not set in xgb.DMatrix", {
  expect_null(dim(dtrain))
})

attr(bst, "feature_names") <- NULL
test_that("feature names don't fail if attributes not set in xgb.DMatrix", {
   expect_equal(xgb.importance(model = bst)$Feature[1], "28")
})
