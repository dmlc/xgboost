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
    data = xgb.DMatrix(dtx, label = dty),
    params = xgb.params(
      objective = "binary:logistic",
      eval_metric = "auc",
      nthread = n_threads
    ),
    nrounds = 100,
    early_stopping_rounds = 3,
    verbose = 0,
    evals = list(train = xgb.DMatrix(dtx, label = dty))
  )

  expect_true(TRUE)  # should not crash
})

test_that("early stopping xgb.cv works", {
  set.seed(11)
  expect_output(
    {
      cv <- xgb.cv(
        c(params, list(learning_rate = 0.3)),
        dtrain,
        nfold = 5,
        nrounds = 20,
        early_stopping_rounds = 3,
        maximize = FALSE
      )
    },
    "Stopping. Best iteration"
  )
  expect_false(is.null(cv$early_stop$best_iteration))
  expect_lt(cv$early_stop$best_iteration, 19)
  # the best error is min error:
  expect_true(cv$evaluation_log[, test_error_mean[cv$early_stop$best_iteration] == min(test_error_mean)])
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
        data = xgb.DMatrix(as.matrix(iris[, -5]), label = lb),
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
