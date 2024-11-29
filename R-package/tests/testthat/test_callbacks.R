# More specific testing of callbacks
context("callbacks")

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
train <- agaricus.train
test <- agaricus.test

n_threads <- 2

# add some label noise for early stopping tests
add.noise <- function(label, frac) {
  inoise <- sample(length(label), length(label) * frac)
  label[inoise] <- !label[inoise]
  label
}
set.seed(11)
ltrain <- add.noise(train$label, 0.2)
ltest <- add.noise(test$label, 0.2)
dtrain <- xgb.DMatrix(train$data, label = ltrain, nthread = n_threads)
dtest <- xgb.DMatrix(test$data, label = ltest, nthread = n_threads)
evals <- list(train = dtrain, test = dtest)


err <- function(label, pr) sum((pr > 0.5) != label) / length(label)

param <- list(objective = "binary:logistic", eval_metric = "error",
              max_depth = 2, nthread = n_threads)


test_that("xgb.cb.print.evaluation works as expected for xgb.train", {
  logs1 <- capture.output({
    model <- xgb.train(
      data = dtrain,
      params = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        max_depth = 2,
        nthread = n_threads
      ),
      nrounds = 10,
      evals = list(train = dtrain, test = dtest),
      callbacks = list(xgb.cb.print.evaluation(period = 1))
    )
  })
  expect_equal(length(logs1), 10)
  expect_true(all(grepl("^\\[\\d{1,2}\\]\ttrain-auc:0\\.\\d+\ttest-auc:0\\.\\d+\\s*$", logs1)))
  lapply(seq(1, 10), function(x) expect_true(grepl(paste0("^\\[", x), logs1[x])))

  logs2 <- capture.output({
    model <- xgb.train(
      data = dtrain,
      params = list(
        objective = "binary:logistic",
        eval_metric = "auc",
        max_depth = 2,
        nthread = n_threads
      ),
      nrounds = 10,
      evals = list(train = dtrain, test = dtest),
      callbacks = list(xgb.cb.print.evaluation(period = 2))
    )
  })
  expect_equal(length(logs2), 6)
  expect_true(all(grepl("^\\[\\d{1,2}\\]\ttrain-auc:0\\.\\d+\ttest-auc:0\\.\\d+\\s*$", logs2)))
  seq_matches <- c(seq(1, 10, 2), 10)
  lapply(seq_along(seq_matches), function(x) expect_true(grepl(paste0("^\\[", seq_matches[x]), logs2[x])))
})

test_that("xgb.cb.evaluation.log works as expected for xgb.train", {
  model <- xgb.train(
    data = dtrain,
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = 2,
      nthread = n_threads
    ),
    nrounds = 10,
    verbose = FALSE,
    evals = list(train = dtrain, test = dtest),
    callbacks = list(xgb.cb.evaluation.log())
  )
  logs <- attributes(model)$evaluation_log

  expect_equal(nrow(logs), 10)
  expect_equal(colnames(logs), c("iter", "train_auc", "test_auc"))
})

param <- list(objective = "binary:logistic", eval_metric = "error",
              max_depth = 4, nthread = n_threads)

test_that("can store evaluation_log without printing", {
  expect_silent(
    bst <- xgb.train(param, dtrain, nrounds = 10, evals = evals, eta = 1, verbose = 0)
  )
  expect_false(is.null(attributes(bst)$evaluation_log))
  expect_false(is.null(attributes(bst)$evaluation_log$train_error))
  expect_lt(attributes(bst)$evaluation_log[, min(train_error)], 0.2)
})

test_that("xgb.cb.reset.parameters works as expected", {

  # fixed eta
  set.seed(111)
  bst0 <- xgb.train(param, dtrain, nrounds = 2, evals = evals, eta = 0.9, verbose = 0)
  expect_false(is.null(attributes(bst0)$evaluation_log))
  expect_false(is.null(attributes(bst0)$evaluation_log$train_error))

  # same eta but re-set as a vector parameter in the callback
  set.seed(111)
  my_par <- list(eta = c(0.9, 0.9))
  bst1 <- xgb.train(param, dtrain, nrounds = 2, evals = evals, verbose = 0,
                    callbacks = list(xgb.cb.reset.parameters(my_par)))
  expect_false(is.null(attributes(bst1)$evaluation_log$train_error))
  expect_equal(attributes(bst0)$evaluation_log$train_error,
               attributes(bst1)$evaluation_log$train_error)

  # same eta but re-set via a function in the callback
  set.seed(111)
  my_par <- list(eta = function(itr, itr_end) 0.9)
  bst2 <- xgb.train(param, dtrain, nrounds = 2, evals = evals, verbose = 0,
                    callbacks = list(xgb.cb.reset.parameters(my_par)))
  expect_false(is.null(attributes(bst2)$evaluation_log$train_error))
  expect_equal(attributes(bst0)$evaluation_log$train_error,
               attributes(bst2)$evaluation_log$train_error)

  # different eta re-set as a vector parameter in the callback
  set.seed(111)
  my_par <- list(eta = c(0.6, 0.5))
  bst3 <- xgb.train(param, dtrain, nrounds = 2, evals = evals, verbose = 0,
                    callbacks = list(xgb.cb.reset.parameters(my_par)))
  expect_false(is.null(attributes(bst3)$evaluation_log$train_error))
  expect_false(all(attributes(bst0)$evaluation_log$train_error == attributes(bst3)$evaluation_log$train_error))

  # resetting multiple parameters at the same time runs with no error
  my_par <- list(eta = c(1., 0.5), gamma = c(1, 2), max_depth = c(4, 8))
  expect_error(
    bst4 <- xgb.train(param, dtrain, nrounds = 2, evals = evals, verbose = 0,
                      callbacks = list(xgb.cb.reset.parameters(my_par)))
  , NA) # NA = no error

  # expect no learning with 0 learning rate
  my_par <- list(eta = c(0., 0.))
  bstX <- xgb.train(param, dtrain, nrounds = 2, evals = evals, verbose = 0,
                    callbacks = list(xgb.cb.reset.parameters(my_par)))
  expect_false(is.null(attributes(bstX)$evaluation_log$train_error))
  er <- unique(attributes(bstX)$evaluation_log$train_error)
  expect_length(er, 1)
  expect_gt(er, 0.4)
})

test_that("xgb.cb.save.model works as expected", {
  files <- c('xgboost_01.json', 'xgboost_02.json', 'xgboost.json')
  files <- unname(sapply(files, function(f) file.path(tempdir(), f)))
  for (f in files) if (file.exists(f)) file.remove(f)

  bst <- xgb.train(param, dtrain, nrounds = 2, evals = evals, eta = 1, verbose = 0,
                   save_period = 1, save_name = file.path(tempdir(), "xgboost_%02d.json"))
  expect_true(file.exists(files[1]))
  expect_true(file.exists(files[2]))
  b1 <- xgb.load(files[1])
  xgb.parameters(b1) <- list(nthread = 2)
  expect_equal(xgb.get.num.boosted.rounds(b1), 1)
  b2 <- xgb.load(files[2])
  xgb.parameters(b2) <- list(nthread = 2)
  expect_equal(xgb.get.num.boosted.rounds(b2), 2)

  xgb.config(b2) <- xgb.config(bst)
  expect_equal(xgb.config(bst), xgb.config(b2))
  expect_equal(xgb.save.raw(bst), xgb.save.raw(b2))

  # save_period = 0 saves the last iteration's model
  bst <- xgb.train(param, dtrain, nrounds = 2, evals = evals, eta = 1, verbose = 0,
                   save_period = 0, save_name = file.path(tempdir(), 'xgboost.json'))
  expect_true(file.exists(files[3]))
  b2 <- xgb.load(files[3])
  xgb.config(b2) <- xgb.config(bst)
  expect_equal(xgb.save.raw(bst), xgb.save.raw(b2))

  for (f in files) if (file.exists(f)) file.remove(f)
})

test_that("early stopping xgb.train works", {
  set.seed(11)
  expect_output(
    bst <- xgb.train(param, dtrain, nrounds = 20, evals = evals, eta = 0.3,
                     early_stopping_rounds = 3, maximize = FALSE)
  , "Stopping. Best iteration")
  expect_false(is.null(xgb.attr(bst, "best_iteration")))
  expect_lt(xgb.attr(bst, "best_iteration"), 19)

  pred <- predict(bst, dtest)
  expect_equal(length(pred), 1611)
  err_pred <- err(ltest, pred)
  err_log <- attributes(bst)$evaluation_log[xgb.attr(bst, "best_iteration") + 1, test_error]
  expect_equal(err_log, err_pred, tolerance = 5e-6)

  set.seed(11)
  expect_silent(
    bst0 <- xgb.train(param, dtrain, nrounds = 20, evals = evals, eta = 0.3,
                      early_stopping_rounds = 3, maximize = FALSE, verbose = 0)
  )
  expect_equal(attributes(bst)$evaluation_log, attributes(bst0)$evaluation_log)

  fname <- file.path(tempdir(), "model.bin")
  xgb.save(bst, fname)
  loaded <- xgb.load(fname)

  expect_false(is.null(xgb.attr(loaded, "best_iteration")))
  expect_equal(xgb.attr(loaded, "best_iteration"), xgb.attr(bst, "best_iteration"))
})

test_that("early stopping using a specific metric works", {
  set.seed(11)
  expect_output(
    bst <- xgb.train(param[-2], dtrain, nrounds = 20, evals = evals, eta = 0.6,
                     eval_metric = "logloss", eval_metric = "auc",
                     callbacks = list(xgb.cb.early.stop(stopping_rounds = 3, maximize = FALSE,
                                                        metric_name = 'test_logloss')))
  , "Stopping. Best iteration")
  expect_false(is.null(xgb.attr(bst, "best_iteration")))
  expect_lt(xgb.attr(bst, "best_iteration"), 19)

  pred <- predict(bst, dtest, iterationrange = c(1, xgb.attr(bst, "best_iteration") + 1))
  expect_equal(length(pred), 1611)
  logloss_pred <- sum(-ltest * log(pred) - (1 - ltest) * log(1 - pred)) / length(ltest)
  logloss_log <- attributes(bst)$evaluation_log[xgb.attr(bst, "best_iteration") + 1, test_logloss]
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

  xgboost::xgb.train(
    data = xgb.DMatrix(dtx, label = dty),
    objective = "binary:logistic",
    eval_metric = "auc",
    nrounds = 100,
    early_stopping_rounds = 3,
    nthread = n_threads,
    evals = list(train = xgb.DMatrix(dtx, label = dty))
  )

  expect_true(TRUE)  # should not crash
})
