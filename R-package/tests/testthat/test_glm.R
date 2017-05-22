context('Test generalized linear models')

require(xgboost)

test_that("glm works", {
  data(agaricus.train, package='xgboost')
  data(agaricus.test, package='xgboost')
  dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
  dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)
  expect_equal(class(dtrain), "xgb.DMatrix")
  expect_equal(class(dtest), "xgb.DMatrix")
  param <- list(objective = "binary:logistic", booster = "gblinear",
                nthread = 2, alpha = 0.0001, lambda = 1)
  watchlist <- list(eval = dtest, train = dtrain)
  num_round <- 2
  bst <- xgb.train(param, dtrain, num_round, watchlist)
  ypred <- predict(bst, dtest)
  expect_equal(length(getinfo(dtest, 'label')), 1611)
})
