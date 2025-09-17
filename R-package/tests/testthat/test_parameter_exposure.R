context('Test model params and call are exposed to R')

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')

dtrain <- xgb.DMatrix(
  agaricus.train$data, label = agaricus.train$label, nthread = 2
)
dtest <- xgb.DMatrix(
  agaricus.test$data, label = agaricus.test$label, nthread = 2
)

bst <- xgb.train(
  data = dtrain,
  verbose = 0,
  nrounds = 10,
  params = xgb.params(
    max_depth = 2,
    learning_rate = 1,
    nthread = 1,
    objective = "binary:logistic"
  )
)

test_that("call is exposed to R", {
  expect_false(is.null(attributes(bst)$call))
  expect_is(attributes(bst)$call, "call")
})

test_that("params is exposed to R", {
  model_params <- attributes(bst)$params
  expect_is(model_params, "list")
  expect_equal(model_params$learning_rate, 1)
  expect_equal(model_params$max_depth, 2)
  expect_equal(model_params$objective, "binary:logistic")
})
