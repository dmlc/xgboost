context('Test model params and call are exposed to R')

require(xgboost)

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')

dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

bst <- xgboost(data = dtrain,
               max.depth = 2,
               eta = 1,
               nround = 10,
               nthread = 1,
               verbose = 0,
               objective = "binary:logistic")

test_that("call is exposed to R", {
  model_call <- attr(bst, "call")
  expect_is(model_call, "call")
})

test_that("params is exposed to R", {
  model_params <- attr(bst, "params")

  expect_is(model_params, "list")

  expect_equal(model_params$eta, 1)
  expect_equal(model_params$max.depth, 2)
  expect_equal(model_params$objective, "binary:logistic")
})
