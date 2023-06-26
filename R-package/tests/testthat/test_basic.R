context("basic functions")

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
train <- agaricus.train
test <- agaricus.test
set.seed(1994)

# disable some tests for Win32
windows_flag <- .Platform$OS.type == "windows" &&
               .Machine$sizeof.pointer != 8
solaris_flag <- (Sys.info()['sysname'] == "SunOS")

test_that("train and predict binary classification", {
  nrounds <- 2
  expect_output(
    bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
                  eta = 1, nthread = 2, nrounds = nrounds, objective = "binary:logistic",
                  eval_metric = "error")
  , "train-error")
  expect_equal(class(bst), "xgb.Booster")
  expect_equal(bst$niter, nrounds)
  expect_false(is.null(bst$evaluation_log))
  expect_equal(nrow(bst$evaluation_log), nrounds)
  expect_lt(bst$evaluation_log[, min(train_error)], 0.03)

  pred <- predict(bst, test$data)
  expect_length(pred, 1611)

  pred1 <- predict(bst, train$data, ntreelimit = 1)
  expect_length(pred1, 6513)
  err_pred1 <- sum((pred1 > 0.5) != train$label) / length(train$label)
  err_log <- bst$evaluation_log[1, train_error]
  expect_lt(abs(err_pred1 - err_log), 10e-6)

  pred2 <- predict(bst, train$data, iterationrange = c(1, 2))
  expect_length(pred1, 6513)
  expect_equal(pred1, pred2)
})

test_that("parameter validation works", {
  p <- list(foo = "bar")
  nrounds <- 1
  set.seed(1994)

  d <- cbind(
    x1 = rnorm(10),
    x2 = rnorm(10),
    x3 = rnorm(10))
  y <- d[, "x1"] + d[, "x2"]^2 +
    ifelse(d[, "x3"] > .5, d[, "x3"]^2, 2^d[, "x3"]) +
    rnorm(10)
  dtrain <- xgb.DMatrix(data = d, info = list(label = y))

  correct <- function() {
    params <- list(max_depth = 2, booster = "dart",
                   rate_drop = 0.5, one_drop = TRUE,
                   objective = "reg:squarederror")
    xgb.train(params = params, data = dtrain, nrounds = nrounds)
  }
  expect_silent(correct())
  incorrect <- function() {
    params <- list(max_depth = 2, booster = "dart",
                   rate_drop = 0.5, one_drop = TRUE,
                   objective = "reg:squarederror",
                   foo = "bar", bar = "foo")
    output <- capture.output(
      xgb.train(params = params, data = dtrain, nrounds = nrounds))
    print(output)
  }
  expect_output(incorrect(), '\\\\"bar\\\\", \\\\"foo\\\\"')
})


test_that("dart prediction works", {
  nrounds <- 32
  set.seed(1994)

  d <- cbind(
    x1 = rnorm(100),
    x2 = rnorm(100),
    x3 = rnorm(100))
  y <- d[, "x1"] + d[, "x2"]^2 +
    ifelse(d[, "x3"] > .5, d[, "x3"]^2, 2^d[, "x3"]) +
    rnorm(100)

  set.seed(1994)
  booster_by_xgboost <- xgboost(data = d, label = y, max_depth = 2, booster = "dart",
                                rate_drop = 0.5, one_drop = TRUE,
                                eta = 1, nthread = 2, nrounds = nrounds, objective = "reg:squarederror")
  pred_by_xgboost_0 <- predict(booster_by_xgboost, newdata = d, ntreelimit = 0)
  pred_by_xgboost_1 <- predict(booster_by_xgboost, newdata = d, ntreelimit = nrounds)
  expect_true(all(matrix(pred_by_xgboost_0, byrow = TRUE) == matrix(pred_by_xgboost_1, byrow = TRUE)))

  pred_by_xgboost_2 <- predict(booster_by_xgboost, newdata = d, training = TRUE)
  expect_false(all(matrix(pred_by_xgboost_0, byrow = TRUE) == matrix(pred_by_xgboost_2, byrow = TRUE)))

  set.seed(1994)
  dtrain <- xgb.DMatrix(data = d, info = list(label = y))
  booster_by_train <- xgb.train(params = list(
                                    booster = "dart",
                                    max_depth = 2,
                                    eta = 1,
                                    rate_drop = 0.5,
                                    one_drop = TRUE,
                                    nthread = 1,
                                    tree_method = "exact",
                                    objective = "reg:squarederror"
                                ),
                                data = dtrain,
                                nrounds = nrounds
                                )
  pred_by_train_0 <- predict(booster_by_train, newdata = dtrain, ntreelimit = 0)
  pred_by_train_1 <- predict(booster_by_train, newdata = dtrain, ntreelimit = nrounds)
  pred_by_train_2 <- predict(booster_by_train, newdata = dtrain, training = TRUE)

  expect_true(all(matrix(pred_by_train_0, byrow = TRUE) == matrix(pred_by_xgboost_0, byrow = TRUE)))
  expect_true(all(matrix(pred_by_train_1, byrow = TRUE) == matrix(pred_by_xgboost_1, byrow = TRUE)))
  expect_true(all(matrix(pred_by_train_2, byrow = TRUE) == matrix(pred_by_xgboost_2, byrow = TRUE)))
})

test_that("train and predict softprob", {
  lb <- as.numeric(iris$Species) - 1
  set.seed(11)
  expect_output(
    bst <- xgboost(data = as.matrix(iris[, -5]), label = lb,
                   max_depth = 3, eta = 0.5, nthread = 2, nrounds = 5,
                   objective = "multi:softprob", num_class = 3, eval_metric = "merror")
  , "train-merror")
  expect_false(is.null(bst$evaluation_log))
  expect_lt(bst$evaluation_log[, min(train_merror)], 0.025)
  expect_equal(bst$niter * 3, xgb.ntree(bst))
  pred <- predict(bst, as.matrix(iris[, -5]))
  expect_length(pred, nrow(iris) * 3)
  # row sums add up to total probability of 1:
  expect_equal(rowSums(matrix(pred, ncol = 3, byrow = TRUE)), rep(1, nrow(iris)), tolerance = 1e-7)
  # manually calculate error at the last iteration:
  mpred <- predict(bst, as.matrix(iris[, -5]), reshape = TRUE)
  expect_equal(as.numeric(t(mpred)), pred)
  pred_labels <- max.col(mpred) - 1
  err <- sum(pred_labels != lb) / length(lb)
  expect_equal(bst$evaluation_log[5, train_merror], err, tolerance = 5e-6)
  # manually calculate error at the 1st iteration:
  mpred <- predict(bst, as.matrix(iris[, -5]), reshape = TRUE, ntreelimit = 1)
  pred_labels <- max.col(mpred) - 1
  err <- sum(pred_labels != lb) / length(lb)
  expect_equal(bst$evaluation_log[1, train_merror], err, tolerance = 5e-6)

  mpred1 <- predict(bst, as.matrix(iris[, -5]), reshape = TRUE, iterationrange = c(1, 2))
  expect_equal(mpred, mpred1)

  d <- cbind(
    x1 = rnorm(100),
    x2 = rnorm(100),
    x3 = rnorm(100)
  )
  y <- sample.int(10, 100, replace = TRUE) - 1
  dtrain <- xgb.DMatrix(data = d, info = list(label = y))
  booster <- xgb.train(
    params = list(tree_method = "hist"), data = dtrain, nrounds = 4, num_class = 10,
    objective = "multi:softprob"
  )
  predt <- predict(booster, as.matrix(d), reshape = TRUE, strict_shape = FALSE)
  expect_equal(ncol(predt), 10)
  expect_equal(rowSums(predt), rep(1, 100), tolerance = 1e-7)
})

test_that("train and predict softmax", {
  lb <- as.numeric(iris$Species) - 1
  set.seed(11)
  expect_output(
    bst <- xgboost(data = as.matrix(iris[, -5]), label = lb,
                   max_depth = 3, eta = 0.5, nthread = 2, nrounds = 5,
                   objective = "multi:softmax", num_class = 3, eval_metric = "merror")
  , "train-merror")
  expect_false(is.null(bst$evaluation_log))
  expect_lt(bst$evaluation_log[, min(train_merror)], 0.025)
  expect_equal(bst$niter * 3, xgb.ntree(bst))

  pred <- predict(bst, as.matrix(iris[, -5]))
  expect_length(pred, nrow(iris))
  err <- sum(pred != lb) / length(lb)
  expect_equal(bst$evaluation_log[5, train_merror], err, tolerance = 5e-6)
})

test_that("train and predict RF", {
  set.seed(11)
  lb <- train$label
  # single iteration
  bst <- xgboost(data = train$data, label = lb, max_depth = 5,
                 nthread = 2, nrounds = 1, objective = "binary:logistic", eval_metric = "error",
                 num_parallel_tree = 20, subsample = 0.6, colsample_bytree = 0.1)
  expect_equal(bst$niter, 1)
  expect_equal(xgb.ntree(bst), 20)

  pred <- predict(bst, train$data)
  pred_err <- sum((pred > 0.5) != lb) / length(lb)
  expect_lt(abs(bst$evaluation_log[1, train_error] - pred_err), 10e-6)
  #expect_lt(pred_err, 0.03)

  pred <- predict(bst, train$data, ntreelimit = 20)
  pred_err_20 <- sum((pred > 0.5) != lb) / length(lb)
  expect_equal(pred_err_20, pred_err)

  pred1 <- predict(bst, train$data, iterationrange = c(1, 2))
  expect_equal(pred, pred1)
})

test_that("train and predict RF with softprob", {
  lb <- as.numeric(iris$Species) - 1
  nrounds <- 15
  set.seed(11)
  bst <- xgboost(data = as.matrix(iris[, -5]), label = lb,
                 max_depth = 3, eta = 0.9, nthread = 2, nrounds = nrounds,
                 objective = "multi:softprob", eval_metric = "merror",
                 num_class = 3, verbose = 0,
                 num_parallel_tree = 4, subsample = 0.5, colsample_bytree = 0.5)
  expect_equal(bst$niter, 15)
  expect_equal(xgb.ntree(bst), 15 * 3 * 4)
  # predict for all iterations:
  pred <- predict(bst, as.matrix(iris[, -5]), reshape = TRUE)
  expect_equal(dim(pred), c(nrow(iris), 3))
  pred_labels <- max.col(pred) - 1
  err <- sum(pred_labels != lb) / length(lb)
  expect_equal(bst$evaluation_log[nrounds, train_merror], err, tolerance = 5e-6)
  # predict for 7 iterations and adjust for 4 parallel trees per iteration
  pred <- predict(bst, as.matrix(iris[, -5]), reshape = TRUE, ntreelimit = 7 * 4)
  err <- sum((max.col(pred) - 1) != lb) / length(lb)
  expect_equal(bst$evaluation_log[7, train_merror], err, tolerance = 5e-6)
})

test_that("use of multiple eval metrics works", {
  expect_output(
    bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
                   eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic",
                   eval_metric = 'error', eval_metric = 'auc', eval_metric = "logloss")
  , "train-error.*train-auc.*train-logloss")
  expect_false(is.null(bst$evaluation_log))
  expect_equal(dim(bst$evaluation_log), c(2, 4))
  expect_equal(colnames(bst$evaluation_log), c("iter", "train_error", "train_auc", "train_logloss"))
  expect_output(
    bst2 <- xgboost(data = train$data, label = train$label, max_depth = 2,
                    eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic",
                    eval_metric = list("error", "auc", "logloss"))
  , "train-error.*train-auc.*train-logloss")
  expect_false(is.null(bst2$evaluation_log))
  expect_equal(dim(bst2$evaluation_log), c(2, 4))
  expect_equal(colnames(bst2$evaluation_log), c("iter", "train_error", "train_auc", "train_logloss"))
})


test_that("training continuation works", {
  dtrain <- xgb.DMatrix(train$data, label = train$label)
  watchlist <- list(train = dtrain)
  param <- list(objective = "binary:logistic", max_depth = 2, eta = 1, nthread = 2)

  # for the reference, use 4 iterations at once:
  set.seed(11)
  bst <- xgb.train(param, dtrain, nrounds = 4, watchlist, verbose = 0)
  # first two iterations:
  set.seed(11)
  bst1 <- xgb.train(param, dtrain, nrounds = 2, watchlist, verbose = 0)
  # continue for two more:
  bst2 <- xgb.train(param, dtrain, nrounds = 2, watchlist, verbose = 0, xgb_model = bst1)
  if (!windows_flag && !solaris_flag)
    expect_equal(bst$raw, bst2$raw)
  expect_false(is.null(bst2$evaluation_log))
  expect_equal(dim(bst2$evaluation_log), c(4, 2))
  expect_equal(bst2$evaluation_log, bst$evaluation_log)
  # test continuing from raw model data
  bst2 <- xgb.train(param, dtrain, nrounds = 2, watchlist, verbose = 0, xgb_model = bst1$raw)
  if (!windows_flag && !solaris_flag)
    expect_equal(bst$raw, bst2$raw)
  expect_equal(dim(bst2$evaluation_log), c(2, 2))
  # test continuing from a model in file
  xgb.save(bst1, "xgboost.json")
  bst2 <- xgb.train(param, dtrain, nrounds = 2, watchlist, verbose = 0, xgb_model = "xgboost.json")
  if (!windows_flag && !solaris_flag)
    expect_equal(bst$raw, bst2$raw)
  expect_equal(dim(bst2$evaluation_log), c(2, 2))
  file.remove("xgboost.json")
})

test_that("model serialization works", {
  out_path <- "model_serialization"
  dtrain <- xgb.DMatrix(train$data, label = train$label)
  watchlist <- list(train = dtrain)
  param <- list(objective = "binary:logistic")
  booster <- xgb.train(param, dtrain, nrounds = 4, watchlist)
  raw <- xgb.serialize(booster)
  saveRDS(raw, out_path)
  raw <- readRDS(out_path)

  loaded <- xgb.unserialize(raw)
  raw_from_loaded <- xgb.serialize(loaded)
  expect_equal(raw, raw_from_loaded)
  file.remove(out_path)
})

test_that("xgb.cv works", {
  set.seed(11)
  expect_output(
    cv <- xgb.cv(data = train$data, label = train$label, max_depth = 2, nfold = 5,
                 eta = 1., nthread = 2, nrounds = 2, objective = "binary:logistic",
                 eval_metric = "error", verbose = TRUE)
  , "train-error:")
  expect_is(cv, 'xgb.cv.synchronous')
  expect_false(is.null(cv$evaluation_log))
  expect_lt(cv$evaluation_log[, min(test_error_mean)], 0.03)
  expect_lt(cv$evaluation_log[, min(test_error_std)], 0.008)
  expect_equal(cv$niter, 2)
  expect_false(is.null(cv$folds) && is.list(cv$folds))
  expect_length(cv$folds, 5)
  expect_false(is.null(cv$params) && is.list(cv$params))
  expect_false(is.null(cv$callbacks))
  expect_false(is.null(cv$call))
})

test_that("xgb.cv works with stratified folds", {
  dtrain <- xgb.DMatrix(train$data, label = train$label)
  set.seed(314159)
  cv <- xgb.cv(data = dtrain, max_depth = 2, nfold = 5,
               eta = 1., nthread = 2, nrounds = 2, objective = "binary:logistic",
               verbose = TRUE, stratified = FALSE)
  set.seed(314159)
  cv2 <- xgb.cv(data = dtrain, max_depth = 2, nfold = 5,
                eta = 1., nthread = 2, nrounds = 2, objective = "binary:logistic",
                verbose = TRUE, stratified = TRUE)
  # Stratified folds should result in a different evaluation logs
  expect_true(all(cv$evaluation_log[, test_logloss_mean] != cv2$evaluation_log[, test_logloss_mean]))
})

test_that("train and predict with non-strict classes", {
  # standard dense matrix input
  train_dense <- as.matrix(train$data)
  bst <- xgboost(data = train_dense, label = train$label, max_depth = 2,
                 eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 0)
  pr0 <- predict(bst, train_dense)

  # dense matrix-like input of non-matrix class
  class(train_dense) <- 'shmatrix'
  expect_true(is.matrix(train_dense))
  expect_error(
    bst <- xgboost(data = train_dense, label = train$label, max_depth = 2,
                   eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 0)
    , regexp = NA)
  expect_error(pr <- predict(bst, train_dense), regexp = NA)
  expect_equal(pr0, pr)

  # dense matrix-like input of non-matrix class with some inheritance
  class(train_dense) <- c('pphmatrix', 'shmatrix')
  expect_true(is.matrix(train_dense))
  expect_error(
    bst <- xgboost(data = train_dense, label = train$label, max_depth = 2,
                   eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 0)
    , regexp = NA)
  expect_error(pr <- predict(bst, train_dense), regexp = NA)
  expect_equal(pr0, pr)

  # when someone inherits from xgb.Booster, it should still be possible to use it as xgb.Booster
  class(bst) <- c('super.Booster', 'xgb.Booster')
  expect_error(pr <- predict(bst, train_dense), regexp = NA)
  expect_equal(pr0, pr)
})

test_that("max_delta_step works", {
  dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
  watchlist <- list(train = dtrain)
  param <- list(objective = "binary:logistic", eval_metric = "logloss", max_depth = 2, nthread = 2, eta = 0.5)
  nrounds <- 5
  # model with no restriction on max_delta_step
  bst1 <- xgb.train(param, dtrain, nrounds, watchlist, verbose = 1)
  # model with restricted max_delta_step
  bst2 <- xgb.train(param, dtrain, nrounds, watchlist, verbose = 1, max_delta_step = 1)
  # the no-restriction model is expected to have consistently lower loss during the initial iterations
  expect_true(all(bst1$evaluation_log$train_logloss < bst2$evaluation_log$train_logloss))
  expect_lt(mean(bst1$evaluation_log$train_logloss) / mean(bst2$evaluation_log$train_logloss), 0.8)
})

test_that("colsample_bytree works", {
  # Randomly generate data matrix by sampling from uniform distribution [-1, 1]
  set.seed(1)
  train_x <- matrix(runif(1000, min = -1, max = 1), ncol = 100)
  train_y <- as.numeric(rowSums(train_x) > 0)
  test_x <- matrix(runif(1000, min = -1, max = 1), ncol = 100)
  test_y <- as.numeric(rowSums(test_x) > 0)
  colnames(train_x) <- paste0("Feature_", sprintf("%03d", 1:100))
  colnames(test_x) <- paste0("Feature_", sprintf("%03d", 1:100))
  dtrain <- xgb.DMatrix(train_x, label = train_y)
  dtest <- xgb.DMatrix(test_x, label = test_y)
  watchlist <- list(train = dtrain, eval = dtest)
  ## Use colsample_bytree = 0.01, so that roughly one out of 100 features is chosen for
  ## each tree
  param <- list(max_depth = 2, eta = 0, nthread = 2,
                colsample_bytree = 0.01, objective = "binary:logistic",
                eval_metric = "auc")
  set.seed(2)
  bst <- xgb.train(param, dtrain, nrounds = 100, watchlist, verbose = 0)
  xgb.importance(model = bst)
  # If colsample_bytree works properly, a variety of features should be used
  # in the 100 trees
  expect_gte(nrow(xgb.importance(model = bst)), 30)
})

test_that("Configuration works", {
  bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
                 eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic",
                 eval_metric = 'error', eval_metric = 'auc', eval_metric = "logloss")
  config <- xgb.config(bst)
  xgb.config(bst) <- config
  reloaded_config <- xgb.config(bst)
  expect_equal(config, reloaded_config)
})

test_that("strict_shape works", {
  n_rounds <- 2

  test_strict_shape <- function(bst, X, n_groups) {
    predt <- predict(bst, X, strict_shape = TRUE)
    margin <- predict(bst, X, outputmargin = TRUE, strict_shape = TRUE)
    contri <- predict(bst, X, predcontrib = TRUE, strict_shape = TRUE)
    interact <- predict(bst, X, predinteraction = TRUE, strict_shape = TRUE)
    leaf <- predict(bst, X, predleaf = TRUE, strict_shape = TRUE)

    n_rows <- nrow(X)
    n_cols <- ncol(X)

    expect_equal(dim(predt), c(n_groups, n_rows))
    expect_equal(dim(margin), c(n_groups, n_rows))
    expect_equal(dim(contri), c(n_cols + 1, n_groups, n_rows))
    expect_equal(dim(interact), c(n_cols + 1, n_cols + 1, n_groups, n_rows))
    expect_equal(dim(leaf), c(1, n_groups, n_rounds, n_rows))

    if (n_groups != 1) {
      for (g in seq_len(n_groups)) {
        expect_lt(max(abs(colSums(contri[, g, ]) - margin[g, ])), 1e-5)
      }
    }
  }

  test_iris <- function() {
    y <- as.numeric(iris$Species) - 1
    X <- as.matrix(iris[, -5])

    bst <- xgboost(data = X, label = y,
                   max_depth = 2, nrounds = n_rounds,
                   objective = "multi:softprob", num_class = 3, eval_metric = "merror")

    test_strict_shape(bst, X, 3)
  }


  test_agaricus <- function() {
    data(agaricus.train, package = 'xgboost')
    X <- agaricus.train$data
    y <- agaricus.train$label

    bst <- xgboost(data = X, label = y, max_depth = 2,
                   nrounds = n_rounds, objective = "binary:logistic",
                   eval_metric = 'error', eval_metric = 'auc', eval_metric = "logloss")

    test_strict_shape(bst, X, 1)
  }

  test_iris()
  test_agaricus()
})

test_that("'predict' accepts CSR data", {
  X <- agaricus.train$data
  y <- agaricus.train$label
  x_csc <- as(X[1L, , drop = FALSE], "CsparseMatrix")
  x_csr <- as(x_csc, "RsparseMatrix")
  x_spv <- as(x_csc, "sparseVector")
  bst <- xgboost(data = X, label = y, objective = "binary:logistic",
                 nrounds = 5L, verbose = FALSE)
  p_csc <- predict(bst, x_csc)
  p_csr <- predict(bst, x_csr)
  p_spv <- predict(bst, x_spv)
  expect_equal(p_csc, p_csr)
  expect_equal(p_csc, p_spv)
})
