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

params <- xgb.params(
  objective = "binary:logistic", eval_metric = "error",
  max_depth = 2, nthread = n_threads
)


test_that("xgb.cb.print.evaluation works as expected for xgb.train", {
  logs1 <- capture.output({
    model <- xgb.train(
      data = dtrain,
      params = xgb.params(
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
      params = xgb.params(
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

test_that("xgb.cb.print.evaluation works as expected for xgb.cv", {
  logs1 <- capture.output({
    model <- xgb.cv(
      data = dtrain,
      params = xgb.params(
        objective = "binary:logistic",
        eval_metric = "auc",
        max_depth = 2,
        nthread = n_threads
      ),
      nrounds = 10,
      nfold = 3,
      callbacks = list(xgb.cb.print.evaluation(period = 1, showsd = TRUE))
    )
  })
  expect_equal(length(logs1), 10)
  expect_true(all(grepl("^\\[\\d{1,2}\\]\ttrain-auc:0\\.\\d+±0\\.\\d+\ttest-auc:0\\.\\d+±0\\.\\d+\\s*$", logs1)))
  lapply(seq(1, 10), function(x) expect_true(grepl(paste0("^\\[", x), logs1[x])))

  logs2 <- capture.output({
    model <- xgb.cv(
      data = dtrain,
      params = xgb.params(
        objective = "binary:logistic",
        eval_metric = "auc",
        max_depth = 2,
        nthread = n_threads
      ),
      nrounds = 10,
      nfold = 3,
      callbacks = list(xgb.cb.print.evaluation(period = 2, showsd = TRUE))
    )
  })
  expect_equal(length(logs2), 6)
  expect_true(all(grepl("^\\[\\d{1,2}\\]\ttrain-auc:0\\.\\d+±0\\.\\d+\ttest-auc:0\\.\\d+±0\\.\\d+\\s*$", logs2)))
  seq_matches <- c(seq(1, 10, 2), 10)
  lapply(seq_along(seq_matches), function(x) expect_true(grepl(paste0("^\\[", seq_matches[x]), logs2[x])))
})

test_that("xgb.cb.evaluation.log works as expected for xgb.train", {
  model <- xgb.train(
    data = dtrain,
    params = xgb.params(
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

test_that("xgb.cb.evaluation.log works as expected for xgb.cv", {
  model <- xgb.cv(
    data = dtrain,
    params = xgb.params(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = 2,
      nthread = n_threads
    ),
    nrounds = 10,
    verbose = FALSE,
    nfold = 3,
    callbacks = list(xgb.cb.evaluation.log())
  )
  logs <- model$evaluation_log

  expect_equal(nrow(logs), 10)
  expect_equal(
    colnames(logs),
    c("iter", "train_auc_mean", "train_auc_std", "test_auc_mean", "test_auc_std")
  )
})


params <- xgb.params(
  objective = "binary:logistic", eval_metric = "error",
  max_depth = 4, nthread = n_threads
)

test_that("can store evaluation_log without printing", {
  expect_silent(
    bst <- xgb.train(params, dtrain, nrounds = 10, evals = evals, verbose = 0)
  )
  expect_false(is.null(attributes(bst)$evaluation_log))
  expect_false(is.null(attributes(bst)$evaluation_log$train_error))
  expect_lt(attributes(bst)$evaluation_log[, min(train_error)], 0.2)
})

test_that("xgb.cb.reset.parameters works as expected", {

  # fixed learning_rate
  params <- c(params, list(learning_rate = 0.9))
  set.seed(111)
  bst0 <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0)
  expect_false(is.null(attributes(bst0)$evaluation_log))
  expect_false(is.null(attributes(bst0)$evaluation_log$train_error))

  # same learning_rate but re-set as a vector parameter in the callback
  set.seed(111)
  my_par <- list(learning_rate = c(0.9, 0.9))
  bst1 <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0,
                    callbacks = list(xgb.cb.reset.parameters(my_par)))
  expect_false(is.null(attributes(bst1)$evaluation_log$train_error))
  expect_equal(attributes(bst0)$evaluation_log$train_error,
               attributes(bst1)$evaluation_log$train_error)

  # same learning_rate but re-set via a function in the callback
  set.seed(111)
  my_par <- list(learning_rate = function(itr, itr_end) 0.9)
  bst2 <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0,
                    callbacks = list(xgb.cb.reset.parameters(my_par)))
  expect_false(is.null(attributes(bst2)$evaluation_log$train_error))
  expect_equal(attributes(bst0)$evaluation_log$train_error,
               attributes(bst2)$evaluation_log$train_error)

  # different learning_rate re-set as a vector parameter in the callback
  set.seed(111)
  my_par <- list(learning_rate = c(0.6, 0.5))
  bst3 <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0,
                    callbacks = list(xgb.cb.reset.parameters(my_par)))
  expect_false(is.null(attributes(bst3)$evaluation_log$train_error))
  expect_false(all(attributes(bst0)$evaluation_log$train_error == attributes(bst3)$evaluation_log$train_error))

  # resetting multiple parameters at the same time runs with no error
  my_par <- list(learning_rate = c(1., 0.5), min_split_loss = c(1, 2), max_depth = c(4, 8))
  expect_error(
    bst4 <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0,
                      callbacks = list(xgb.cb.reset.parameters(my_par)))
  , NA) # NA = no error
  # CV works as well
  expect_error(
    bst4 <- xgb.cv(params, dtrain, nfold = 2, nrounds = 2, verbose = 0,
                   callbacks = list(xgb.cb.reset.parameters(my_par)))
  , NA) # NA = no error

  # expect no learning with 0 learning rate
  my_par <- list(learning_rate = c(0., 0.))
  bstX <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0,
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

  bst <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0,
                   save_period = 1, save_name = file.path(tempdir(), "xgboost_%02d.json"))
  expect_true(file.exists(files[1]))
  expect_true(file.exists(files[2]))
  b1 <- xgb.load(files[1])
  xgb.model.parameters(b1) <- list(nthread = 2)
  expect_equal(xgb.get.num.boosted.rounds(b1), 1)
  b2 <- xgb.load(files[2])
  xgb.model.parameters(b2) <- list(nthread = 2)
  expect_equal(xgb.get.num.boosted.rounds(b2), 2)

  xgb.config(b2) <- xgb.config(bst)
  expect_equal(xgb.config(bst), xgb.config(b2))
  expect_equal(xgb.save.raw(bst), xgb.save.raw(b2))

  # save_period = 0 saves the last iteration's model
  bst <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0,
                   save_period = 0, save_name = file.path(tempdir(), 'xgboost.json'))
  expect_true(file.exists(files[3]))
  b2 <- xgb.load(files[3])
  xgb.config(b2) <- xgb.config(bst)
  expect_equal(xgb.save.raw(bst), xgb.save.raw(b2))

  for (f in files) if (file.exists(f)) file.remove(f)
})

test_that("early stopping prints the complete evaluation at the best iteration", {
  # Deliberately give the metrics different best iterations and stop later than either.
  metrics <- cbind(
    "test-auc" = c(0.6, 0.8, 0.7, 0.65, 0.6),
    "test-rmse" = c(1.4, 1.3, 1.1, 1.2, 1.3)
  )
  for (metric_name in list(NULL, "test_rmse", "test-auc")) {
    maximize <- identical(metric_name, "test-auc")
    best <- if (maximize) 2L else 3L
    for (begin_iteration in c(1L, 7L)) {
      for (verbose in c(TRUE, FALSE)) {
        cb <- xgb.cb.early.stop(2, maximize, metric_name, verbose)
        cb$f_before_training(cb$env, list(), NULL, NULL, begin_iteration, begin_iteration + 4L)
        stopped <- logical()
        output <- capture.output({
          for (i in seq_len(best + 2L)) {
            iteration <- begin_iteration + i - 1L
            stopped[i] <- cb$f_after_iter(cb$env, list(), NULL, NULL, iteration, metrics[i, ])
          }
        })
        result <- cb$f_after_training(cb$env, list(), NULL, NULL, iteration, metrics[i, ], NULL)
        expect_equal(which(stopped), best + 2L)
        expect_equal(result$best_iteration, begin_iteration + best - 1L)
        expect_equal(unname(result$best_score), metrics[best, if (maximize) 1L else 2L])
        expect_true(result$stopped_by_max_rounds)
        if (verbose) {
          best_line <- output[match("Stopping. Best iteration:", output) + 1L]
          expect_equal(
            best_line,
            sprintf(
              "[%d]\ttest-auc:%f\ttest-rmse:%f",
              result$best_iteration, metrics[best, 1L], metrics[best, 2L]
            )
          )
        } else {
          expect_length(output, 0L)
        }
      }
    }
  }
})

test_that("early stopping prints the stored best state when the metric never improves", {
  for (begin_iteration in c(1L, 7L)) {
    cb <- xgb.cb.early.stop(2, maximize = FALSE, metric_name = "test_rmse")
    cb$f_before_training(cb$env, list(), NULL, NULL, begin_iteration, begin_iteration + 4L)
    stopped <- logical()
    output <- capture.output({
      for (i in seq_len(3L)) {
        iteration <- begin_iteration + i - 1L
        metrics <- c("train-rmse" = i, "test-rmse" = Inf)
        if (i == 3L) {
          expect_null(cb$env$best_msg)
        }
        stopped[i] <- cb$f_after_iter(cb$env, list(), NULL, NULL, iteration, metrics)
      }
    })
    result <- cb$f_after_training(cb$env, list(), NULL, NULL, iteration, metrics, NULL)
    expect_equal(stopped, c(FALSE, FALSE, TRUE))
    expect_equal(result$best_iteration, begin_iteration)
    expect_equal(result$best_score, Inf)
    expect_true(result$stopped_by_max_rounds)
    expect_equal(cb$env$best_msg, sprintf("[%d]\ttest-rmse:Inf", begin_iteration))
    expect_equal(output[match("Stopping. Best iteration:", output) + 1L], cb$env$best_msg)
  }
})

test_that("early stopping prints standard deviations from the best CV iteration", {
  cb <- xgb.cb.early.stop(2)
  cb$f_before_training(cb$env, list(), NULL, NULL, 1L, 4L)
  output <- capture.output({
    for (i in seq_len(4L)) {
      values <- c(2, 1, 3, 4)[i]
      metrics <- rbind("train-rmse" = c(values, values + 0.2), "test-rmse" = c(values, values + 0.4))
      stopped <- cb$f_after_iter(cb$env, list(), NULL, NULL, i, metrics)
    }
  })
  expect_true(stopped)
  expect_equal(cb$env$best_iteration, 2L)
  expect_equal(
    output[match("Stopping. Best iteration:", output) + 1L],
    sprintf("[2]\ttrain-rmse:1.100000\u00b1%f\ttest-rmse:1.200000\u00b1%f", sd(c(1, 1.2)), sd(c(1, 1.4)))
  )
})

test_that("verbose RMSE early stopping reports the best evaluation (issue #12643)", {
  set.seed(1)
  x <- matrix(rnorm(400), ncol = 2)
  y <- x[, 1] + rnorm(200)
  dtr <- xgb.DMatrix(x[1:100, ], label = y[1:100], nthread = 1)
  dte <- xgb.DMatrix(x[101:200, ], label = y[101:200], nthread = 1)
  output <- capture.output({
    bst <- xgb.train(
      list(eta = 0.2, max_depth = 6, nthread = 1), dtr, nrounds = 50,
      evals = list(test = dte), early_stopping_rounds = 10, verbose = TRUE
    )
  })
  log <- attributes(bst)$evaluation_log
  best <- which.min(log$test_rmse)
  expect_equal(xgb.attr(bst, "best_iteration"), best - 1L)
  expect_equal(attributes(bst)$early_stop$best_iteration, best)
  expect_lt(best, nrow(log))
  expect_equal(
    output[match("Stopping. Best iteration:", output) + 1L],
    sprintf("[%d]\ttest-rmse:%f", log$iter[best], log$test_rmse[best])
  )
})

test_that("verbose AUC early stopping preserves attributes and save_best", {
  for (save_best in c(FALSE, TRUE)) {
    output <- capture.output({
      bst <- xgb.train(
        list(objective = "binary:logistic", eval_metric = "auc", eta = 0, base_score = 0.5, nthread = n_threads),
        dtrain, nrounds = 5, evals = list(test = dtest),
        callbacks = list(xgb.cb.early.stop(2, maximize = TRUE, save_best = save_best))
      )
    })
    log <- attributes(bst)$evaluation_log
    best <- which.max(log$test_auc)
    expect_equal(best, 1L)
    expect_equal(nrow(log), 3L)
    expect_equal(xgb.attr(bst, "best_iteration"), best - 1L)
    expect_equal(xgb.attr(bst, "best_score"), log$test_auc[best])
    expect_equal(attributes(bst)$early_stop$best_iteration, best)
    expect_equal(xgb.get.num.boosted.rounds(bst), if (save_best) best else nrow(log))
    expect_equal(
      output[match("Stopping. Best iteration:", output) + 1L],
      sprintf("[%d]\ttest-auc:%f", log$iter[best], log$test_auc[best])
    )
  }
})

test_that("verbose early stopping uses the previous best evaluation when resuming", {
  p <- list(
    objective = "binary:logistic", eval_metric = "auc", eval_metric = "error",
    eta = 0, base_score = 0.5, nthread = n_threads
  )
  bst <- xgb.train(
    p, dtrain, nrounds = 2, evals = list(test = dtest), verbose = FALSE,
    callbacks = list(xgb.cb.early.stop(3, maximize = TRUE, metric_name = "test-auc", verbose = FALSE))
  )
  log <- attributes(bst)$evaluation_log
  without_history <- xgb.copy.Booster(bst)
  attr(without_history, "evaluation_log") <- NULL
  # R serialization uses the native memory serialization hooks; xgb.save.raw saves model bytes.
  for (previous_model in list(bst, without_history, xgb.save.raw(bst), unserialize(serialize(bst, NULL)))) {
    keep_history <- !is.null(attr(previous_model, "evaluation_log"))
    output <- capture.output({
      resumed <- xgb.train(
        p, dtrain, nrounds = 5, evals = list(test = dtest), xgb_model = previous_model,
        callbacks = list(xgb.cb.early.stop(3, maximize = TRUE, metric_name = "test-auc"))
      )
    })
    expect_equal(xgb.attr(resumed, "best_iteration"), 0L)
    expect_equal(xgb.attr(resumed, "best_score"), log$test_auc[1L])
    expect_equal(attributes(resumed)$early_stop$best_iteration, 1L)
    expect_equal(xgb.get.num.boosted.rounds(resumed), 4L)
    expected <- sprintf("[1]\ttest-auc:%f", log$test_auc[1L])
    if (keep_history) {
      expected <- paste0(expected, sprintf("\ttest-error:%f", log$test_error[1L]))
    }
    expect_equal(output[match("Stopping. Best iteration:", output) + 1L], expected)
  }
})

test_that("early stopping replaces the previous best message after a new best when resuming", {
  p <- list(objective = "binary:logistic", eta = 0, base_score = 0.5, nthread = n_threads)
  bst <- xgb.train(
    p, dtrain, nrounds = 2, evals = evals, verbose = FALSE, maximize = FALSE,
    custom_metric = function(preds, data) list(metric = "rmse", value = 2),
    callbacks = list(xgb.cb.early.stop(2, metric_name = "test_rmse", verbose = FALSE))
  )
  expect_equal(xgb.attr(bst, "best_iteration"), 0L)
  expect_equal(xgb.attr(bst, "best_score"), 2)
  for (previous_model in list(bst, xgb.save.raw(bst))) {
    values <- cbind(train = c(1.2, 1.0, 0.9, 0.8), test = c(1.0, 0.8, 0.9, 1.0))
    calls <- 0L
    metric <- function(preds, data) {
      calls <<- calls + 1L
      round <- (calls - 1L) %/% 2L + 1L
      dataset <- (calls - 1L) %% 2L + 1L
      return(list(metric = "rmse", value = values[round, dataset]))
    }
    cb <- xgb.cb.early.stop(2, metric_name = "test_rmse")
    output <- capture.output({
      resumed <- xgb.train(
        p, dtrain, nrounds = 4, evals = evals, xgb_model = previous_model,
        custom_metric = metric, maximize = FALSE, callbacks = list(cb)
      )
    })
    expect_equal(calls, 8L)
    expect_equal(xgb.attr(resumed, "best_iteration"), 3L)
    expect_equal(xgb.attr(resumed, "best_score"), 0.8)
    expect_equal(attributes(resumed)$early_stop$best_iteration, 4L)
    expect_true(attributes(resumed)$early_stop$stopped_by_max_rounds)
    expect_equal(xgb.get.num.boosted.rounds(resumed), 6L)
    expect_equal(cb$env$best_msg, "[4]\ttrain-rmse:1.000000\ttest-rmse:0.800000")
    expect_equal(output[match("Stopping. Best iteration:", output) + 1L], cb$env$best_msg)
  }
})

test_that("verbose early stopping leaves training without a stop unchanged", {
  cb <- xgb.cb.early.stop(2)
  cb$f_before_training(cb$env, list(), NULL, NULL, 1L, 3L)
  output <- capture.output({
    for (i in seq_len(3L)) {
      stopped <- cb$f_after_iter(cb$env, list(), NULL, NULL, i, c("test-rmse" = 4 - i))
      expect_false(stopped)
    }
  })
  expect_false(any(grepl("Stopping. Best iteration:", output, fixed = TRUE)))
  expect_equal(cb$env$best_iteration, 3L)
  expect_equal(unname(cb$env$best_score), 1)
  expect_false(cb$env$stopped_by_max_rounds)
})

test_that("early stopping xgb.train works", {
  params <- c(params, list(learning_rate = 0.3))
  set.seed(11)
  expect_output(
    bst <- xgb.train(params, dtrain, nrounds = 20, evals = evals,
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
    bst0 <- xgb.train(params, dtrain, nrounds = 20, evals = evals,
                      early_stopping_rounds = 3, maximize = FALSE, verbose = 0)
  )
  expect_equal(attributes(bst)$evaluation_log, attributes(bst0)$evaluation_log)

  fname <- file.path(tempdir(), "model.ubj")
  xgb.save(bst, fname)
  loaded <- xgb.load(fname)

  expect_false(is.null(xgb.attr(loaded, "best_iteration")))
  expect_equal(xgb.attr(loaded, "best_iteration"), xgb.attr(bst, "best_iteration"))
})

test_that("early stopping using a specific metric works", {
  set.seed(11)
  expect_output(
    bst <- xgb.train(
      c(
        within(params, rm("eval_metric")),
        list(
          learning_rate = 0.6,
          eval_metric = "logloss",
          eval_metric = "auc"
        )
      ),
      dtrain,
      nrounds = 20,
      evals = evals,
      callbacks = list(
        xgb.cb.early.stop(stopping_rounds = 3, maximize = FALSE, metric_name = 'test_logloss')
      )
    )
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

  xgb.train(
    data = xgb.DMatrix(dtx, label = dty, nthread = 1),
    params = xgb.params(
      objective = "binary:logistic",
      eval_metric = "auc",
      nthread = n_threads
    ),
    nrounds = 100,
    early_stopping_rounds = 3,
    verbose = 0,
    evals = list(train = xgb.DMatrix(dtx, label = dty, nthread = 1))
  )

  expect_true(TRUE)  # should not crash
})

test_that("early stopping xgb.cv works", {
  set.seed(11)
  output <- capture.output({
    cv <- xgb.cv(
      c(params, list(learning_rate = 0.3)),
      dtrain,
      nfold = 5,
      nrounds = 20,
      early_stopping_rounds = 3,
      maximize = FALSE
    )
  })
  expect_false(is.null(cv$early_stop$best_iteration))
  expect_lt(cv$early_stop$best_iteration, 19)
  # the best error is min error:
  expect_true(cv$evaluation_log[, test_error_mean[cv$early_stop$best_iteration] == min(test_error_mean)])
  best_row <- cv$evaluation_log[cv$early_stop$best_iteration]
  expect_equal(
    output[match("Stopping. Best iteration:", output) + 1L],
    sprintf(
      "[%d]\ttrain-error:%f\u00b1%f\ttest-error:%f\u00b1%f",
      best_row$iter, best_row$train_error_mean, best_row$train_error_std,
      best_row$test_error_mean, best_row$test_error_std
    )
  )
})

test_that("prediction in xgb.cv works", {
  params <- c(params, list(learning_rate = 0.5))
  set.seed(11)
  nrounds <- 4
  cv <- xgb.cv(params, dtrain, nfold = 5, nrounds = nrounds, prediction = TRUE, verbose = 0)
  expect_false(is.null(cv$evaluation_log))
  expect_false(is.null(cv$cv_predict$pred))
  expect_length(cv$cv_predict$pred, nrow(train$data))
  err_pred <- mean(sapply(cv$folds, function(f) mean(err(ltrain[f], cv$cv_predict$pred[f]))))
  err_log <- cv$evaluation_log[nrounds, test_error_mean]
  expect_equal(err_pred, err_log, tolerance = 1e-6)

  # save CV models
  set.seed(11)
  cvx <- xgb.cv(params, dtrain, nfold = 5, nrounds = nrounds, prediction = TRUE, verbose = 0,
                callbacks = list(xgb.cb.cv.predict(save_models = TRUE)))
  expect_equal(cv$evaluation_log, cvx$evaluation_log)
  expect_length(cvx$cv_predict$models, 5)
  expect_true(all(sapply(cvx$cv_predict$models, class) == 'xgb.Booster'))
})

test_that("prediction in xgb.cv works for gblinear too", {
  set.seed(11)
  p <- xgb.params(
    booster = 'gblinear',
    objective = "reg:logistic",
    learning_rate = 0.5,
    nthread = n_threads
  )
  cv <- xgb.cv(p, dtrain, nfold = 5, nrounds = 2, prediction = TRUE, verbose = 0)
  expect_false(is.null(cv$evaluation_log))
  expect_false(is.null(cv$cv_predict$pred))
  expect_length(cv$cv_predict$pred, nrow(train$data))
})

test_that("prediction in early-stopping xgb.cv works", {
  params <- c(params, list(learning_rate = 0.1, base_score = 0.5))
  set.seed(11)
  expect_output(
    cv <- xgb.cv(params, dtrain, nfold = 5, nrounds = 20,
                 early_stopping_rounds = 5, maximize = FALSE, stratified = FALSE,
                 prediction = TRUE, verbose = TRUE)
  , "Stopping. Best iteration")

  expect_false(is.null(cv$early_stop$best_iteration))
  expect_lt(cv$early_stop$best_iteration, 19)
  expect_false(is.null(cv$evaluation_log))
  expect_false(is.null(cv$cv_predict$pred))
  expect_length(cv$cv_predict$pred, nrow(train$data))

  err_pred <- mean(sapply(cv$folds, function(f) mean(err(ltrain[f], cv$cv_predict$pred[f]))))
  err_log <- cv$evaluation_log[cv$early_stop$best_iteration, test_error_mean]
  expect_equal(err_pred, err_log, tolerance = 1e-6)
  err_log_last <- cv$evaluation_log[cv$niter, test_error_mean]
  expect_gt(abs(err_pred - err_log_last), 1e-4)
})

test_that("prediction in xgb.cv for softprob works", {
  lb <- as.numeric(iris$Species) - 1
  set.seed(11)
  expect_warning(
    {
      cv <- xgb.cv(
        data = xgb.DMatrix(as.matrix(iris[, -5]), label = lb, nthread = 1),
        nfold = 4,
        nrounds = 5,
        params = xgb.params(
          objective = "multi:softprob",
          num_class = 3,
          learning_rate = 0.5,
          max_depth = 3,
          nthread = n_threads,
          subsample = 0.8,
          min_split_loss = 2
        ),
        verbose = 0,
        prediction = TRUE
      )
    },
    NA
  )
  expect_false(is.null(cv$cv_predict$pred))
  expect_equal(dim(cv$cv_predict$pred), c(nrow(iris), 3))
  expect_lt(diff(range(rowSums(cv$cv_predict$pred))), 1e-6)
})

test_that("prediction in xgb.cv works for multi-quantile", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  cv <- xgb.cv(
    data = dm,
    params = xgb.params(
      objective = "reg:quantileerror",
      quantile_alpha = c(0.1, 0.2, 0.5, 0.8, 0.9),
      nthread = 1
    ),
    nrounds = 5,
    nfold = 3,
    prediction = TRUE,
    verbose = 0
  )
  expect_equal(dim(cv$cv_predict$pred), c(nrow(x), 5))
})

test_that("prediction in xgb.cv works for multi-output", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = cbind(y, -y), nthread = 1)
  cv <- xgb.cv(
    data = dm,
    params = xgb.params(
      tree_method = "hist",
      multi_strategy = "multi_output_tree",
      objective = "reg:squarederror",
      nthread = n_threads
    ),
    nrounds = 5,
    nfold = 3,
    prediction = TRUE,
    verbose = 0
  )
  expect_equal(dim(cv$cv_predict$pred), c(nrow(x), 2))
})

test_that("prediction in xgb.cv works for multi-quantile", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  cv <- xgb.cv(
    data = dm,
    params = xgb.params(
      objective = "reg:quantileerror",
      quantile_alpha = c(0.1, 0.2, 0.5, 0.8, 0.9),
      nthread = 1
    ),
    nrounds = 5,
    nfold = 3,
    prediction = TRUE,
    verbose = 0
  )
  expect_equal(dim(cv$cv_predict$pred), c(nrow(x), 5))
})

test_that("prediction in xgb.cv works for multi-output", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = cbind(y, -y), nthread = 1)
  cv <- xgb.cv(
    data = dm,
    params = xgb.params(
      tree_method = "hist",
      multi_strategy = "multi_output_tree",
      objective = "reg:squarederror",
      nthread = n_threads
    ),
    nrounds = 5,
    nfold = 3,
    prediction = TRUE,
    verbose = 0
  )
  expect_equal(dim(cv$cv_predict$pred), c(nrow(x), 2))
})
