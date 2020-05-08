context('Test generalized linear models')

require(xgboost)

test_that("gblinear works", {
  data(agaricus.train, package='xgboost')
  data(agaricus.test, package='xgboost')
  dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
  dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

  param <- list(objective = "binary:logistic", booster = "gblinear",
                nthread = 2, eta = 0.8, alpha = 0.0001, lambda = 0.0001)
  watchlist <- list(eval = dtest, train = dtrain)

  n <- 5         # iterations
  ERR_UL <- 0.005 # upper limit for the test set error
  VERB <- 0      # chatterbox switch

  param$updater = 'shotgun'
  bst <- xgb.train(param, dtrain, n, watchlist, verbose = VERB, feature_selector = 'shuffle')
  ypred <- predict(bst, dtest)
  expect_equal(length(getinfo(dtest, 'label')), 1611)
  expect_lt(bst$evaluation_log$eval_error[n], ERR_UL)

  bst <- xgb.train(param, dtrain, n, watchlist, verbose = VERB, feature_selector = 'cyclic',
                   callbacks = list(cb.gblinear.history()))
  expect_lt(bst$evaluation_log$eval_error[n], ERR_UL)
  h <- xgb.gblinear.history(bst)
  expect_equal(dim(h), c(n, ncol(dtrain) + 1))
  expect_is(h, "matrix")

  param$updater = 'coord_descent'
  bst <- xgb.train(param, dtrain, n, watchlist, verbose = VERB, feature_selector = 'cyclic')
  expect_lt(bst$evaluation_log$eval_error[n], ERR_UL)

  bst <- xgb.train(param, dtrain, n, watchlist, verbose = VERB, feature_selector = 'shuffle')
  expect_lt(bst$evaluation_log$eval_error[n], ERR_UL)

  bst <- xgb.train(param, dtrain, 2, watchlist, verbose = VERB, feature_selector = 'greedy')
  expect_lt(bst$evaluation_log$eval_error[2], ERR_UL)

  bst <- xgb.train(param, dtrain, n, watchlist, verbose = VERB, feature_selector = 'thrifty',
                   top_k = 50, callbacks = list(cb.gblinear.history(sparse = TRUE)))
  expect_lt(bst$evaluation_log$eval_error[n], ERR_UL)
  h <- xgb.gblinear.history(bst)
  expect_equal(dim(h), c(n, ncol(dtrain) + 1))
  expect_s4_class(h, "dgCMatrix")
})
