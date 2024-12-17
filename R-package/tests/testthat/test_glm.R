context('Test generalized linear models')

n_threads <- 2

test_that("gblinear works", {
  data(agaricus.train, package = 'xgboost')
  data(agaricus.test, package = 'xgboost')
  dtrain <- xgb.DMatrix(
    agaricus.train$data, label = agaricus.train$label, nthread = n_threads
  )
  dtest <- xgb.DMatrix(
    agaricus.test$data, label = agaricus.test$label, nthread = n_threads
  )

  param <- list(objective = "binary:logistic", eval_metric = "error", booster = "gblinear",
                nthread = n_threads, learning_rate = 0.8, reg_alpha = 0.0001, reg_lambda = 0.0001)
  evals <- list(eval = dtest, train = dtrain)

  n <- 5         # iterations
  ERR_UL <- 0.005 # upper limit for the test set error
  VERB <- 0      # chatterbox switch

  param$updater <- 'shotgun'
  bst <- xgb.train(c(param, list(feature_selector = 'shuffle')), dtrain, n, evals, verbose = VERB)
  ypred <- predict(bst, dtest)
  expect_equal(length(getinfo(dtest, 'label')), 1611)
  expect_lt(attributes(bst)$evaluation_log$eval_error[n], ERR_UL)

  bst <- xgb.train(c(param, list(feature_selector = 'cyclic')), dtrain, n, evals, verbose = VERB,
                   callbacks = list(xgb.cb.gblinear.history()))
  expect_lt(attributes(bst)$evaluation_log$eval_error[n], ERR_UL)
  h <- xgb.gblinear.history(bst)
  expect_equal(dim(h), c(n, ncol(dtrain) + 1))
  expect_is(h, "matrix")

  param$updater <- 'coord_descent'
  bst <- xgb.train(c(param, list(feature_selector = 'cyclic')), dtrain, n, evals, verbose = VERB)
  expect_lt(attributes(bst)$evaluation_log$eval_error[n], ERR_UL)

  bst <- xgb.train(c(param, list(feature_selector = 'shuffle')), dtrain, n, evals, verbose = VERB)
  expect_lt(attributes(bst)$evaluation_log$eval_error[n], ERR_UL)

  bst <- xgb.train(c(param, list(feature_selector = 'greedy')), dtrain, 2, evals, verbose = VERB)
  expect_lt(attributes(bst)$evaluation_log$eval_error[2], ERR_UL)

  bst <- xgb.train(c(param, list(feature_selector = 'thrifty', top_k = 50)), dtrain, n, evals, verbose = VERB,
                   callbacks = list(xgb.cb.gblinear.history(sparse = TRUE)))
  expect_lt(attributes(bst)$evaluation_log$eval_error[n], ERR_UL)
  h <- xgb.gblinear.history(bst)
  expect_equal(dim(h), c(n, ncol(dtrain) + 1))
  expect_s4_class(h, "dgCMatrix")
})

test_that("gblinear early stopping works", {
  data(agaricus.train, package = 'xgboost')
  data(agaricus.test, package = 'xgboost')
  dtrain <- xgb.DMatrix(
    agaricus.train$data, label = agaricus.train$label, nthread = n_threads
  )
  dtest <- xgb.DMatrix(
    agaricus.test$data, label = agaricus.test$label, nthread = n_threads
  )

  param <- xgb.params(
    objective = "binary:logistic", eval_metric = "error", booster = "gblinear",
    nthread = n_threads, learning_rate = 0.8, reg_alpha = 0.0001, reg_lambda = 0.0001,
    updater = "coord_descent"
  )

  es_round <- 1
  n <- 10
  booster <- xgb.train(
    param, dtrain, nrounds = n, evals = list(eval = dtest, train = dtrain),
    early_stopping_rounds = es_round, verbose = 0
  )
  expect_equal(xgb.attr(booster, "best_iteration"), 4)
  predt_es <- predict(booster, dtrain)

  n <- xgb.attr(booster, "best_iteration") + es_round + 1
  booster <- xgb.train(
    param, dtrain, nrounds = n, evals = list(eval = dtest, train = dtrain),
    early_stopping_rounds = es_round, verbose = 0
  )
  predt <- predict(booster, dtrain)
  expect_equal(predt_es, predt)
})
