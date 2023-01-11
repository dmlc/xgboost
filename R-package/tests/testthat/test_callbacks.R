# More specific testing of callbacks
context("callbacks")

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
train <- agaricus.train
test <- agaricus.test

# add some label noise for early stopping tests
add.noise <- function(label, frac) {
  inoise <- sample(length(label), length(label) * frac)
  label[inoise] <- !label[inoise]
  label
}
set.seed(11)
ltrain <- add.noise(train$label, 0.2)
ltest <- add.noise(test$label, 0.2)
dtrain <- xgb.DMatrix(train$data, label = ltrain)
dtest <- xgb.DMatrix(test$data, label = ltest)
watchlist <- list(train = dtrain, test = dtest)


err <- function(label, pr) sum((pr > 0.5) != label) / length(label)

param <- list(objective = "binary:logistic", eval_metric = "error",
              max_depth = 2, nthread = 2)


test_that("cb.print.evaluation works as expected", {

  bst_evaluation <- c('train-auc' = 0.9, 'test-auc' = 0.8)
  bst_evaluation_err <- NULL
  begin_iteration <- 1
  end_iteration <- 7

  f0 <- cb.print.evaluation(period = 0)
  f1 <- cb.print.evaluation(period = 1)
  f5 <- cb.print.evaluation(period = 5)

  expect_false(is.null(attr(f1, 'call')))
  expect_equal(attr(f1, 'name'), 'cb.print.evaluation')

  iteration <- 1
  expect_silent(f0())
  expect_output(f1(), "\\[1\\]\ttrain-auc:0.900000\ttest-auc:0.800000")
  expect_output(f5(), "\\[1\\]\ttrain-auc:0.900000\ttest-auc:0.800000")
  expect_null(f1())

  iteration <- 2
  expect_output(f1(), "\\[2\\]\ttrain-auc:0.900000\ttest-auc:0.800000")
  expect_silent(f5())

  iteration <- 7
  expect_output(f1(), "\\[7\\]\ttrain-auc:0.900000\ttest-auc:0.800000")
  expect_output(f5(), "\\[7\\]\ttrain-auc:0.900000\ttest-auc:0.800000")

  bst_evaluation_err  <- c('train-auc' = 0.1, 'test-auc' = 0.2)
  expect_output(f1(), "\\[7\\]\ttrain-auc:0.900000\\+0.100000\ttest-auc:0.800000\\+0.200000")
})

test_that("cb.evaluation.log works as expected", {

  bst_evaluation <- c('train-auc' = 0.9, 'test-auc' = 0.8)
  bst_evaluation_err <- NULL

  evaluation_log <- list()
  f <- cb.evaluation.log()

  expect_false(is.null(attr(f, 'call')))
  expect_equal(attr(f, 'name'), 'cb.evaluation.log')

  iteration <- 1
  expect_silent(f())
  expect_equal(evaluation_log,
               list(c(iter = 1, bst_evaluation)))
  iteration <- 2
  expect_silent(f())
  expect_equal(evaluation_log,
               list(c(iter = 1, bst_evaluation), c(iter = 2, bst_evaluation)))
  expect_silent(f(finalize = TRUE))
  expect_equal(evaluation_log,
               data.table::data.table(iter = 1:2, train_auc = c(0.9, 0.9), test_auc = c(0.8, 0.8)))

  bst_evaluation_err  <- c('train-auc' = 0.1, 'test-auc' = 0.2)
  evaluation_log <- list()
  f <- cb.evaluation.log()

  iteration <- 1
  expect_silent(f())
  expect_equal(evaluation_log,
               list(c(iter = 1, c(bst_evaluation, bst_evaluation_err))))
  iteration <- 2
  expect_silent(f())
  expect_equal(evaluation_log,
               list(c(iter = 1, c(bst_evaluation, bst_evaluation_err)),
                    c(iter = 2, c(bst_evaluation, bst_evaluation_err))))
  expect_silent(f(finalize = TRUE))
  expect_equal(evaluation_log,
               data.table::data.table(iter = 1:2,
                          train_auc_mean = c(0.9, 0.9), train_auc_std = c(0.1, 0.1),
                          test_auc_mean = c(0.8, 0.8), test_auc_std = c(0.2, 0.2)))
})


param <- list(objective = "binary:logistic", eval_metric = "error",
              max_depth = 4, nthread = 2)

test_that("can store evaluation_log without printing", {
  expect_silent(
    bst <- xgb.train(param, dtrain, nrounds = 10, watchlist, eta = 1, verbose = 0)
  )
  expect_false(is.null(bst$evaluation_log))
  expect_false(is.null(bst$evaluation_log$train_error))
  expect_lt(bst$evaluation_log[, min(train_error)], 0.2)
})

test_that("cb.reset.parameters works as expected", {

  # fixed eta
  set.seed(111)
  bst0 <- xgb.train(param, dtrain, nrounds = 2, watchlist, eta = 0.9, verbose = 0)
  expect_false(is.null(bst0$evaluation_log))
  expect_false(is.null(bst0$evaluation_log$train_error))

  # same eta but re-set as a vector parameter in the callback
  set.seed(111)
  my_par <- list(eta = c(0.9, 0.9))
  bst1 <- xgb.train(param, dtrain, nrounds = 2, watchlist, verbose = 0,
                    callbacks = list(cb.reset.parameters(my_par)))
  expect_false(is.null(bst1$evaluation_log$train_error))
  expect_equal(bst0$evaluation_log$train_error,
               bst1$evaluation_log$train_error)

  # same eta but re-set via a function in the callback
  set.seed(111)
  my_par <- list(eta = function(itr, itr_end) 0.9)
  bst2 <- xgb.train(param, dtrain, nrounds = 2, watchlist, verbose = 0,
                    callbacks = list(cb.reset.parameters(my_par)))
  expect_false(is.null(bst2$evaluation_log$train_error))
  expect_equal(bst0$evaluation_log$train_error,
               bst2$evaluation_log$train_error)

  # different eta re-set as a vector parameter in the callback
  set.seed(111)
  my_par <- list(eta = c(0.6, 0.5))
  bst3 <- xgb.train(param, dtrain, nrounds = 2, watchlist, verbose = 0,
                    callbacks = list(cb.reset.parameters(my_par)))
  expect_false(is.null(bst3$evaluation_log$train_error))
  expect_false(all(bst0$evaluation_log$train_error == bst3$evaluation_log$train_error))

  # resetting multiple parameters at the same time runs with no error
  my_par <- list(eta = c(1., 0.5), gamma = c(1, 2), max_depth = c(4, 8))
  expect_error(
    bst4 <- xgb.train(param, dtrain, nrounds = 2, watchlist, verbose = 0,
                      callbacks = list(cb.reset.parameters(my_par)))
  , NA) # NA = no error
  # CV works as well
  expect_error(
    bst4 <- xgb.cv(param, dtrain, nfold = 2, nrounds = 2, verbose = 0,
                   callbacks = list(cb.reset.parameters(my_par)))
  , NA) # NA = no error

  # expect no learning with 0 learning rate
  my_par <- list(eta = c(0., 0.))
  bstX <- xgb.train(param, dtrain, nrounds = 2, watchlist, verbose = 0,
                    callbacks = list(cb.reset.parameters(my_par)))
  expect_false(is.null(bstX$evaluation_log$train_error))
  er <- unique(bstX$evaluation_log$train_error)
  expect_length(er, 1)
  expect_gt(er, 0.4)
})

test_that("cb.save.model works as expected", {
  files <- c('xgboost_01.json', 'xgboost_02.json', 'xgboost.json')
  for (f in files) if (file.exists(f)) file.remove(f)

  bst <- xgb.train(param, dtrain, nrounds = 2, watchlist, eta = 1, verbose = 0,
                   save_period = 1, save_name = "xgboost_%02d.json")
  expect_true(file.exists('xgboost_01.json'))
  expect_true(file.exists('xgboost_02.json'))
  b1 <- xgb.load('xgboost_01.json')
  expect_equal(xgb.ntree(b1), 1)
  b2 <- xgb.load('xgboost_02.json')
  expect_equal(xgb.ntree(b2), 2)

  xgb.config(b2) <- xgb.config(bst)
  expect_equal(xgb.config(bst), xgb.config(b2))
  expect_equal(bst$raw, b2$raw)

  # save_period = 0 saves the last iteration's model
  bst <- xgb.train(param, dtrain, nrounds = 2, watchlist, eta = 1, verbose = 0,
                   save_period = 0, save_name = 'xgboost.json')
  expect_true(file.exists('xgboost.json'))
  b2 <- xgb.load('xgboost.json')
  xgb.config(b2) <- xgb.config(bst)
  expect_equal(bst$raw, b2$raw)

  for (f in files) if (file.exists(f)) file.remove(f)
})

test_that("early stopping xgb.train works", {
  set.seed(11)
  expect_output(
    bst <- xgb.train(param, dtrain, nrounds = 20, watchlist, eta = 0.3,
                     early_stopping_rounds = 3, maximize = FALSE)
  , "Stopping. Best iteration")
  expect_false(is.null(bst$best_iteration))
  expect_lt(bst$best_iteration, 19)
  expect_equal(bst$best_iteration, bst$best_ntreelimit)

  pred <- predict(bst, dtest)
  expect_equal(length(pred), 1611)
  err_pred <- err(ltest, pred)
  err_log <- bst$evaluation_log[bst$best_iteration, test_error]
  expect_equal(err_log, err_pred, tolerance = 5e-6)

  set.seed(11)
  expect_silent(
    bst0 <- xgb.train(param, dtrain, nrounds = 20, watchlist, eta = 0.3,
                      early_stopping_rounds = 3, maximize = FALSE, verbose = 0)
  )
  expect_equal(bst$evaluation_log, bst0$evaluation_log)

  xgb.save(bst, "model.bin")
  loaded <- xgb.load("model.bin")

  expect_false(is.null(loaded$best_iteration))
  expect_equal(loaded$best_iteration, bst$best_ntreelimit)
  expect_equal(loaded$best_ntreelimit, bst$best_ntreelimit)

  file.remove("model.bin")
})

test_that("early stopping using a specific metric works", {
  set.seed(11)
  expect_output(
    bst <- xgb.train(param[-2], dtrain, nrounds = 20, watchlist, eta = 0.6,
                     eval_metric = "logloss", eval_metric = "auc",
                     callbacks = list(cb.early.stop(stopping_rounds = 3, maximize = FALSE,
                                                    metric_name = 'test_logloss')))
  , "Stopping. Best iteration")
  expect_false(is.null(bst$best_iteration))
  expect_lt(bst$best_iteration, 19)
  expect_equal(bst$best_iteration, bst$best_ntreelimit)

  pred <- predict(bst, dtest, ntreelimit = bst$best_ntreelimit)
  expect_equal(length(pred), 1611)
  logloss_pred <- sum(-ltest * log(pred) - (1 - ltest) * log(1 - pred)) / length(ltest)
  logloss_log <- bst$evaluation_log[bst$best_iteration, test_logloss]
  expect_equal(logloss_log, logloss_pred, tolerance = 1e-5)
})

test_that("early stopping works with titanic", {
  if (!requireNamespace("titanic")) {
    testthat::skip("Optional testing dependency 'titanic' not found.")
  }
  # This test was inspired by https://github.com/dmlc/xgboost/issues/5935
  # It catches possible issues on noLD R
  titanic <- titanic::titanic_train
  titanic$Pclass <-  as.factor(titanic$Pclass)
  dtx <- model.matrix(~ 0 + ., data = titanic[, c("Pclass", "Sex")])
  dty <- titanic$Survived

  xgboost::xgboost(
    data = dtx,
    label = dty,
    objective = "binary:logistic",
    eval_metric = "auc",
    nrounds = 100,
    early_stopping_rounds = 3
  )

  expect_true(TRUE)  # should not crash
})

test_that("early stopping xgb.cv works", {
  set.seed(11)
  expect_output(
    cv <- xgb.cv(param, dtrain, nfold = 5, eta = 0.3, nrounds = 20,
                 early_stopping_rounds = 3, maximize = FALSE)
  , "Stopping. Best iteration")
  expect_false(is.null(cv$best_iteration))
  expect_lt(cv$best_iteration, 19)
  expect_equal(cv$best_iteration, cv$best_ntreelimit)
  # the best error is min error:
  expect_true(cv$evaluation_log[, test_error_mean[cv$best_iteration] == min(test_error_mean)])
})

test_that("prediction in xgb.cv works", {
  set.seed(11)
  nrounds <- 4
  cv <- xgb.cv(param, dtrain, nfold = 5, eta = 0.5, nrounds = nrounds, prediction = TRUE, verbose = 0)
  expect_false(is.null(cv$evaluation_log))
  expect_false(is.null(cv$pred))
  expect_length(cv$pred, nrow(train$data))
  err_pred <- mean(sapply(cv$folds, function(f) mean(err(ltrain[f], cv$pred[f]))))
  err_log <- cv$evaluation_log[nrounds, test_error_mean]
  expect_equal(err_pred, err_log, tolerance = 1e-6)

  # save CV models
  set.seed(11)
  cvx <- xgb.cv(param, dtrain, nfold = 5, eta = 0.5, nrounds = nrounds, prediction = TRUE, verbose = 0,
                callbacks = list(cb.cv.predict(save_models = TRUE)))
  expect_equal(cv$evaluation_log, cvx$evaluation_log)
  expect_length(cvx$models, 5)
  expect_true(all(sapply(cvx$models, class) == 'xgb.Booster'))
})

test_that("prediction in xgb.cv works for gblinear too", {
  set.seed(11)
  p <- list(booster = 'gblinear', objective = "reg:logistic", nthread = 2)
  cv <- xgb.cv(p, dtrain, nfold = 5, eta = 0.5, nrounds = 2, prediction = TRUE, verbose = 0)
  expect_false(is.null(cv$evaluation_log))
  expect_false(is.null(cv$pred))
  expect_length(cv$pred, nrow(train$data))
})

test_that("prediction in early-stopping xgb.cv works", {
  set.seed(11)
  expect_output(
    cv <- xgb.cv(param, dtrain, nfold = 5, eta = 0.1, nrounds = 20,
                 early_stopping_rounds = 5, maximize = FALSE, stratified = FALSE,
                 prediction = TRUE, base_score = 0.5)
  , "Stopping. Best iteration")

  expect_false(is.null(cv$best_iteration))
  expect_lt(cv$best_iteration, 19)
  expect_false(is.null(cv$evaluation_log))
  expect_false(is.null(cv$pred))
  expect_length(cv$pred, nrow(train$data))

  err_pred <- mean(sapply(cv$folds, function(f) mean(err(ltrain[f], cv$pred[f]))))
  err_log <- cv$evaluation_log[cv$best_iteration, test_error_mean]
  expect_equal(err_pred, err_log, tolerance = 1e-6)
  err_log_last <- cv$evaluation_log[cv$niter, test_error_mean]
  expect_gt(abs(err_pred - err_log_last), 1e-4)
})

test_that("prediction in xgb.cv for softprob works", {
  lb <- as.numeric(iris$Species) - 1
  set.seed(11)
  expect_warning(
    cv <- xgb.cv(data = as.matrix(iris[, -5]), label = lb, nfold = 4,
                 eta = 0.5, nrounds = 5, max_depth = 3, nthread = 2,
                 subsample = 0.8, gamma = 2, verbose = 0,
                 prediction = TRUE, objective = "multi:softprob", num_class = 3)
  , NA)
  expect_false(is.null(cv$pred))
  expect_equal(dim(cv$pred), c(nrow(iris), 3))
  expect_lt(diff(range(rowSums(cv$pred))), 1e-6)
})
