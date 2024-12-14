context("basic functions")

data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")
train <- agaricus.train
test <- agaricus.test
set.seed(1994)

# disable some tests for Win32
windows_flag <- .Platform$OS.type == "windows" &&
  .Machine$sizeof.pointer != 8
solaris_flag <- (Sys.info()["sysname"] == "SunOS")
n_threads <- 1


test_that("train and predict binary classification", {
  nrounds <- 2
  expect_output(
    bst <- xgb.train(
      data = xgb.DMatrix(train$data, label = train$label),
      nrounds = nrounds,
      params = xgb.params(
        max_depth = 2,
        learning_rate = 1,
        nthread = n_threads,
        objective = "binary:logistic",
        eval_metric = "error"
      ),
      evals = list(train = xgb.DMatrix(train$data, label = train$label))
    ),
    "train-error"
  )
  expect_equal(class(bst), "xgb.Booster")
  expect_equal(xgb.get.num.boosted.rounds(bst), nrounds)
  expect_false(is.null(attributes(bst)$evaluation_log))
  expect_equal(nrow(attributes(bst)$evaluation_log), nrounds)
  expect_lt(attributes(bst)$evaluation_log[, min(train_error)], 0.03)

  pred <- predict(bst, test$data)
  expect_length(pred, 1611)

  pred1 <- predict(bst, train$data, iterationrange = c(1, 1))
  expect_length(pred1, 6513)
  err_pred1 <- sum((pred1 > 0.5) != train$label) / length(train$label)
  err_log <- attributes(bst)$evaluation_log[1, train_error]
  expect_lt(abs(err_pred1 - err_log), 10e-6)
})

test_that("parameter validation works", {
  p <- list(foo = "bar")
  nrounds <- 1
  set.seed(1994)

  d <- cbind(
    x1 = rnorm(10),
    x2 = rnorm(10),
    x3 = rnorm(10)
  )
  y <- d[, "x1"] + d[, "x2"]^2 +
    ifelse(d[, "x3"] > .5, d[, "x3"]^2, 2^d[, "x3"]) +
    rnorm(10)
  dtrain <- xgb.DMatrix(data = d, label = y, nthread = n_threads)

  correct <- function() {
    params <- list(
      max_depth = 2,
      booster = "dart",
      rate_drop = 0.5,
      one_drop = TRUE,
      nthread = n_threads,
      objective = "reg:squarederror"
    )
    xgb.train(params = params, data = dtrain, nrounds = nrounds)
  }
  expect_silent(correct())
  incorrect <- function() {
    params <- list(
      max_depth = 2,
      booster = "dart",
      rate_drop = 0.5,
      one_drop = TRUE,
      objective = "reg:squarederror",
      nthread = n_threads,
      foo = "bar",
      bar = "foo"
    )
    output <- capture.output(
      xgb.train(params = params, data = dtrain, nrounds = nrounds),
      type = "message"
    )
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
    x3 = rnorm(100)
  )
  y <- d[, "x1"] + d[, "x2"]^2 +
    ifelse(d[, "x3"] > .5, d[, "x3"]^2, 2^d[, "x3"]) +
    rnorm(100)

  set.seed(1994)
  booster_by_xgboost <- xgb.train(
    data = xgb.DMatrix(d, label = y),
    nrounds = nrounds,
    params = xgb.params(
      max_depth = 2,
      booster = "dart",
      rate_drop = 0.5,
      one_drop = TRUE,
      learning_rate = 1,
      nthread = n_threads,
      objective = "reg:squarederror"
    )
  )
  pred_by_xgboost_0 <- predict(booster_by_xgboost, newdata = d, iterationrange = NULL)
  pred_by_xgboost_1 <- predict(booster_by_xgboost, newdata = d, iterationrange = c(1, nrounds))
  expect_true(all(matrix(pred_by_xgboost_0, byrow = TRUE) == matrix(pred_by_xgboost_1, byrow = TRUE)))

  pred_by_xgboost_2 <- predict(booster_by_xgboost, newdata = d, training = TRUE)
  expect_false(all(matrix(pred_by_xgboost_0, byrow = TRUE) == matrix(pred_by_xgboost_2, byrow = TRUE)))

  set.seed(1994)
  dtrain <- xgb.DMatrix(data = d, label = y, nthread = n_threads)
  booster_by_train <- xgb.train(
    params = xgb.params(
      booster = "dart",
      max_depth = 2,
      learning_rate = 1,
      rate_drop = 0.5,
      one_drop = TRUE,
      nthread = n_threads,
      objective = "reg:squarederror"
    ),
    data = dtrain,
    nrounds = nrounds
  )
  pred_by_train_0 <- predict(booster_by_train, newdata = dtrain, iterationrange = NULL)
  pred_by_train_1 <- predict(booster_by_train, newdata = dtrain, iterationrange = c(1, nrounds))
  pred_by_train_2 <- predict(booster_by_train, newdata = dtrain, training = TRUE)

  expect_equal(pred_by_train_0, pred_by_xgboost_0, tolerance = 1e-6)
  expect_equal(pred_by_train_1, pred_by_xgboost_1, tolerance = 1e-6)
  expect_true(all(matrix(pred_by_train_2, byrow = TRUE) == matrix(pred_by_xgboost_2, byrow = TRUE)))
})

test_that("train and predict softprob", {
  lb <- as.numeric(iris$Species) - 1
  set.seed(11)
  expect_output(
    bst <- xgb.train(
      data = xgb.DMatrix(as.matrix(iris[, -5]), label = lb),
      nrounds = 5,
      params = xgb.params(
        max_depth = 3, learning_rate = 0.5, nthread = n_threads,
        objective = "multi:softprob", num_class = 3, eval_metric = "merror"
      ),
      evals = list(train = xgb.DMatrix(as.matrix(iris[, -5]), label = lb))
    ),
    "train-merror"
  )
  expect_false(is.null(attributes(bst)$evaluation_log))
  expect_lt(attributes(bst)$evaluation_log[, min(train_merror)], 0.025)
  expect_equal(xgb.get.num.boosted.rounds(bst), 5)
  pred <- predict(bst, as.matrix(iris[, -5]))
  expect_length(pred, nrow(iris) * 3)
  # row sums add up to total probability of 1:
  expect_equal(rowSums(pred), rep(1, nrow(iris)), tolerance = 1e-7)
  # manually calculate error at the last iteration:
  mpred <- predict(bst, as.matrix(iris[, -5]))
  expect_equal(mpred, pred)
  pred_labels <- max.col(mpred) - 1
  err <- sum(pred_labels != lb) / length(lb)
  expect_equal(attributes(bst)$evaluation_log[5, train_merror], err, tolerance = 5e-6)
  # manually calculate error at the 1st iteration:
  mpred <- predict(bst, as.matrix(iris[, -5]), iterationrange = c(1, 1))
  pred_labels <- max.col(mpred) - 1
  err <- sum(pred_labels != lb) / length(lb)
  expect_equal(attributes(bst)$evaluation_log[1, train_merror], err, tolerance = 5e-6)

  mpred1 <- predict(bst, as.matrix(iris[, -5]), iterationrange = c(1, 1))
  expect_equal(mpred, mpred1)

  d <- cbind(
    x1 = rnorm(100),
    x2 = rnorm(100),
    x3 = rnorm(100)
  )
  y <- sample.int(10, 100, replace = TRUE) - 1
  dtrain <- xgb.DMatrix(data = d, label = y, nthread = n_threads)
  booster <- xgb.train(
    params = xgb.params(
      objective = "multi:softprob",
      num_class = 10,
      tree_method = "hist",
      nthread = n_threads
    ),
    data = dtrain,
    nrounds = 4
  )
  predt <- predict(booster, as.matrix(d), strict_shape = FALSE)
  expect_equal(ncol(predt), 10)
  expect_equal(rowSums(predt), rep(1, 100), tolerance = 1e-7)
})

test_that("train and predict softmax", {
  lb <- as.numeric(iris$Species) - 1
  set.seed(11)
  expect_output(
    bst <- xgb.train(
      data = xgb.DMatrix(as.matrix(iris[, -5]), label = lb),
      nrounds = 5,
      params = xgb.params(
        max_depth = 3, learning_rate = 0.5, nthread = n_threads,
        objective = "multi:softmax", num_class = 3, eval_metric = "merror"
      ),
      evals = list(train = xgb.DMatrix(as.matrix(iris[, -5]), label = lb))
    ),
    "train-merror"
  )
  expect_false(is.null(attributes(bst)$evaluation_log))
  expect_lt(attributes(bst)$evaluation_log[, min(train_merror)], 0.025)
  expect_equal(xgb.get.num.boosted.rounds(bst), 5)

  pred <- predict(bst, as.matrix(iris[, -5]))
  expect_length(pred, nrow(iris))
  err <- sum(pred != lb) / length(lb)
  expect_equal(attributes(bst)$evaluation_log[5, train_merror], err, tolerance = 5e-6)
})

test_that("train and predict RF", {
  set.seed(11)
  lb <- train$label
  # single iteration
  bst <- xgb.train(
    data = xgb.DMatrix(train$data, label = lb),
    nrounds = 1,
    params = xgb.params(
      max_depth = 5,
      nthread = n_threads,
      objective = "binary:logistic", eval_metric = "error",
      num_parallel_tree = 20, subsample = 0.6, colsample_bytree = 0.1
    ),
    evals = list(train = xgb.DMatrix(train$data, label = lb)),
    verbose = 0
  )
  expect_equal(xgb.get.num.boosted.rounds(bst), 1)

  pred <- predict(bst, train$data)
  pred_err <- sum((pred > 0.5) != lb) / length(lb)
  expect_lt(abs(attributes(bst)$evaluation_log[1, train_error] - pred_err), 10e-6)
  # expect_lt(pred_err, 0.03)

  pred <- predict(bst, train$data, iterationrange = c(1, 1))
  pred_err_20 <- sum((pred > 0.5) != lb) / length(lb)
  expect_equal(pred_err_20, pred_err)
})

test_that("train and predict RF with softprob", {
  lb <- as.numeric(iris$Species) - 1
  nrounds <- 15
  set.seed(11)
  bst <- xgb.train(
    data = xgb.DMatrix(as.matrix(iris[, -5]), label = lb),
    nrounds = nrounds,
    verbose = 0,
    params = xgb.params(
      max_depth = 3,
      learning_rate = 0.9,
      nthread = n_threads,
      objective = "multi:softprob",
      eval_metric = "merror",
      num_class = 3,
      num_parallel_tree = 4,
      subsample = 0.5,
      colsample_bytree = 0.5
    ),
    evals = list(train = xgb.DMatrix(as.matrix(iris[, -5]), label = lb))
  )
  expect_equal(xgb.get.num.boosted.rounds(bst), 15)
  # predict for all iterations:
  pred <- predict(bst, as.matrix(iris[, -5]))
  expect_equal(dim(pred), c(nrow(iris), 3))
  pred_labels <- max.col(pred) - 1
  err <- sum(pred_labels != lb) / length(lb)
  expect_equal(attributes(bst)$evaluation_log[nrounds, train_merror], err, tolerance = 5e-6)
  # predict for 7 iterations and adjust for 4 parallel trees per iteration
  pred <- predict(bst, as.matrix(iris[, -5]), iterationrange = c(1, 7))
  err <- sum((max.col(pred) - 1) != lb) / length(lb)
  expect_equal(attributes(bst)$evaluation_log[7, train_merror], err, tolerance = 5e-6)
})

test_that("use of multiple eval metrics works", {
  expect_output(
    bst <- xgb.train(
      data = xgb.DMatrix(train$data, label = train$label),
      nrounds = 2,
      params = list(
        max_depth = 2,
        learning_rate = 1, nthread = n_threads, objective = "binary:logistic",
        eval_metric = "error", eval_metric = "auc", eval_metric = "logloss"
      ),
      evals = list(train = xgb.DMatrix(train$data, label = train$label))
    ),
    "train-error.*train-auc.*train-logloss"
  )
  expect_false(is.null(attributes(bst)$evaluation_log))
  expect_equal(dim(attributes(bst)$evaluation_log), c(2, 4))
  expect_equal(colnames(attributes(bst)$evaluation_log), c("iter", "train_error", "train_auc", "train_logloss"))
  expect_output(
    bst2 <- xgb.train(
      data = xgb.DMatrix(train$data, label = train$label),
      nrounds = 2,
      params = xgb.params(
        max_depth = 2,
        learning_rate = 1,
        nthread = n_threads,
        objective = "binary:logistic",
        eval_metric = list("error", "auc", "logloss")
      ),
      evals = list(train = xgb.DMatrix(train$data, label = train$label))
    ),
    "train-error.*train-auc.*train-logloss"
  )
  expect_false(is.null(attributes(bst2)$evaluation_log))
  expect_equal(dim(attributes(bst2)$evaluation_log), c(2, 4))
  expect_equal(colnames(attributes(bst2)$evaluation_log), c("iter", "train_error", "train_auc", "train_logloss"))
})


test_that("training continuation works", {
  dtrain <- xgb.DMatrix(train$data, label = train$label, nthread = n_threads)
  evals <- list(train = dtrain)
  params <- xgb.params(
    objective = "binary:logistic", max_depth = 2, learning_rate = 1, nthread = n_threads
  )

  # for the reference, use 4 iterations at once:
  set.seed(11)
  bst <- xgb.train(params, dtrain, nrounds = 4, evals = evals, verbose = 0)
  # first two iterations:
  set.seed(11)
  bst1 <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0)
  # continue for two more:
  bst2 <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0, xgb_model = bst1)
  if (!windows_flag && !solaris_flag) {
    expect_equal(xgb.save.raw(bst), xgb.save.raw(bst2))
  }
  expect_false(is.null(attributes(bst2)$evaluation_log))
  expect_equal(dim(attributes(bst2)$evaluation_log), c(4, 2))
  expect_equal(attributes(bst2)$evaluation_log, attributes(bst)$evaluation_log)
  # test continuing from raw model data
  bst2 <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0, xgb_model = xgb.save.raw(bst1))
  if (!windows_flag && !solaris_flag) {
    expect_equal(xgb.save.raw(bst), xgb.save.raw(bst2))
  }
  expect_equal(dim(attributes(bst2)$evaluation_log), c(2, 2))
  # test continuing from a model in file
  fname <- file.path(tempdir(), "xgboost.json")
  xgb.save(bst1, fname)
  bst2 <- xgb.train(params, dtrain, nrounds = 2, evals = evals, verbose = 0, xgb_model = fname)
  if (!windows_flag && !solaris_flag) {
    expect_equal(xgb.save.raw(bst), xgb.save.raw(bst2))
  }
  expect_equal(dim(attributes(bst2)$evaluation_log), c(2, 2))
})

test_that("xgb.cv works", {
  set.seed(11)
  expect_output(
    cv <- xgb.cv(
      data = xgb.DMatrix(train$data, label = train$label),
      nfold = 5,
      nrounds = 2,
      params = xgb.params(
        max_depth = 2,
        learning_rate = 1.,
        nthread = n_threads,
        objective = "binary:logistic",
        eval_metric = "error"
      ),
      verbose = TRUE
    ),
    "train-error:"
  )
  expect_is(cv, "xgb.cv.synchronous")
  expect_false(is.null(cv$evaluation_log))
  expect_lt(cv$evaluation_log[, min(test_error_mean)], 0.03)
  expect_lt(cv$evaluation_log[, min(test_error_std)], 0.0085)
  expect_equal(cv$niter, 2)
  expect_false(is.null(cv$folds) && is.list(cv$folds))
  expect_length(cv$folds, 5)
  expect_false(is.null(cv$params) && is.list(cv$params))
  expect_false(is.null(cv$call))
})

test_that("xgb.cv works with stratified folds", {
  dtrain <- xgb.DMatrix(train$data, label = train$label, nthread = n_threads)
  set.seed(314159)
  cv <- xgb.cv(
    data = dtrain,
    nrounds = 2,
    nfold = 5,
    params = xgb.params(
      max_depth = 2,
      nthread = n_threads,
      objective = "binary:logistic"
    ),
    verbose = FALSE, stratified = FALSE
  )
  set.seed(314159)
  cv2 <- xgb.cv(
    data = dtrain,
    nfold = 5,
    nrounds = 2,
    params = xgb.params(
      max_depth = 2,
      nthread = n_threads,
      objective = "binary:logistic"
    ),
    verbose = FALSE, stratified = TRUE
  )
  # Stratified folds should result in a different evaluation logs
  expect_true(all(cv$evaluation_log[, test_logloss_mean] != cv2$evaluation_log[, test_logloss_mean]))
})

test_that("train and predict with non-strict classes", {
  # standard dense matrix input
  train_dense <- as.matrix(train$data)
  bst <- xgb.train(
    data = xgb.DMatrix(train_dense, label = train$label),
    nrounds = 2,
    params = xgb.params(
      max_depth = 2,
      nthread = n_threads,
      objective = "binary:logistic"
    ),
    verbose = 0
  )
  pr0 <- predict(bst, train_dense)

  # dense matrix-like input of non-matrix class
  class(train_dense) <- "shmatrix"
  expect_true(is.matrix(train_dense))
  expect_error(
    bst <- xgb.train(
      data = xgb.DMatrix(train_dense, label = train$label),
      nrounds = 2,
      params = xgb.params(
        max_depth = 2,
        nthread = n_threads,
        objective = "binary:logistic"
      ),
      verbose = 0
    ),
    regexp = NA
  )
  expect_error(pr <- predict(bst, train_dense), regexp = NA)
  expect_equal(pr0, pr)

  # dense matrix-like input of non-matrix class with some inheritance
  class(train_dense) <- c("pphmatrix", "shmatrix")
  expect_true(is.matrix(train_dense))
  expect_error(
    bst <- xgb.train(
      data = xgb.DMatrix(train_dense, label = train$label),
      nrounds = 2,
      params = xgb.params(
        max_depth = 2,
        nthread = n_threads,
        objective = "binary:logistic"
      ),
      verbose = 0
    ),
    regexp = NA
  )
  expect_error(pr <- predict(bst, train_dense), regexp = NA)
  expect_equal(pr0, pr)

  # when someone inherits from xgb.Booster, it should still be possible to use it as xgb.Booster
  class(bst) <- c("super.Booster", "xgb.Booster")
  expect_error(pr <- predict(bst, train_dense), regexp = NA)
  expect_equal(pr0, pr)
})

test_that("max_delta_step works", {
  dtrain <- xgb.DMatrix(
    agaricus.train$data, label = agaricus.train$label, nthread = n_threads
  )
  evals <- list(train = dtrain)
  params <- xgb.params(
    objective = "binary:logistic", eval_metric = "logloss", max_depth = 2,
    nthread = n_threads,
    learning_rate = 0.5
  )
  nrounds <- 5
  # model with no restriction on max_delta_step
  bst1 <- xgb.train(params, dtrain, nrounds, evals = evals, verbose = 0)
  # model with restricted max_delta_step
  bst2 <- xgb.train(c(params, list(max_delta_step = 1)), dtrain, nrounds, evals = evals, verbose = 0)
  # the no-restriction model is expected to have consistently lower loss during the initial iterations
  expect_true(all(attributes(bst1)$evaluation_log$train_logloss < attributes(bst2)$evaluation_log$train_logloss))
  expect_lt(mean(attributes(bst1)$evaluation_log$train_logloss) / mean(attributes(bst2)$evaluation_log$train_logloss), 0.8)
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
  dtrain <- xgb.DMatrix(train_x, label = train_y, nthread = n_threads)
  dtest <- xgb.DMatrix(test_x, label = test_y, nthread = n_threads)
  evals <- list(train = dtrain, eval = dtest)
  ## Use colsample_bytree = 0.01, so that roughly one out of 100 features is chosen for
  ## each tree
  params <- xgb.params(
    max_depth = 2, learning_rate = 0, nthread = n_threads,
    colsample_bytree = 0.01, objective = "binary:logistic",
    eval_metric = "auc"
  )
  set.seed(2)
  bst <- xgb.train(params, dtrain, nrounds = 100, evals = evals, verbose = 0)
  xgb.importance(model = bst)
  # If colsample_bytree works properly, a variety of features should be used
  # in the 100 trees
  expect_gte(nrow(xgb.importance(model = bst)), 28)
})

test_that("Configuration works", {
  bst <- xgb.train(
    data = xgb.DMatrix(train$data, label = train$label),
    nrounds = 2,
    params = xgb.params(
      max_depth = 2,
      nthread = n_threads,
      objective = "binary:logistic"
    )
  )
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

    expect_equal(dim(predt), c(n_rows, n_groups))
    expect_equal(dim(margin), c(n_rows, n_groups))
    expect_equal(dim(contri), c(n_rows, n_groups, n_cols + 1))
    expect_equal(dim(interact), c(n_rows, n_groups, n_cols + 1, n_cols + 1))
    expect_equal(dim(leaf), c(n_rows, n_rounds, n_groups, 1))

    if (n_groups != 1) {
      for (g in seq_len(n_groups)) {
        expect_lt(max(abs(rowSums(contri[, g, ]) - margin[, g])), 1e-5)
      }

      leaf_no_strict <- predict(bst, X, strict_shape = FALSE, predleaf = TRUE)
      for (g in seq_len(n_groups)) {
        g_mask <- rep(FALSE, n_groups)
        g_mask[g] <- TRUE
        expect_equal(
          leaf[, , g, 1L],
          leaf_no_strict[, g_mask]
        )
      }
    }
  }

  test_iris <- function() {
    y <- as.numeric(iris$Species) - 1
    X <- as.matrix(iris[, -5])

    bst <- xgb.train(
      data = xgb.DMatrix(X, label = y),
      nrounds = n_rounds,
      params = xgb.params(
        max_depth = 2, nthread = n_threads,
        objective = "multi:softprob", num_class = 3
      )
    )

    test_strict_shape(bst, X, 3)
  }


  test_agaricus <- function() {
    data(agaricus.train, package = "xgboost")
    X <- agaricus.train$data
    y <- agaricus.train$label

    bst <- xgb.train(
      data = xgb.DMatrix(X, label = y),
      nrounds = n_rounds,
      params = xgb.params(
        max_depth = 2, nthread = n_threads,
        objective = "binary:logistic"
      )
    )

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
  bst <- xgb.train(
    data = xgb.DMatrix(X, label = y),
    nrounds = 5L, verbose = FALSE,
    params = xgb.params(
      objective = "binary:logistic",
      nthread = n_threads
    )
  )
  p_csc <- predict(bst, x_csc)
  p_csr <- predict(bst, x_csr)
  p_spv <- predict(bst, x_spv)
  expect_equal(p_csc, p_csr)
  expect_equal(p_csc, p_spv)
})

test_that("Quantile regression accepts multiple quantiles", {
  data(mtcars)
  y <- mtcars[, 1]
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(data = x, label = y)
  model <- xgb.train(
    data = dm,
    params = xgb.params(
      objective = "reg:quantileerror",
      tree_method = "exact",
      quantile_alpha = c(0.05, 0.5, 0.95),
      nthread = n_threads
    ),
    nrounds = 15
  )
  pred <- predict(model, x)

  expect_equal(dim(pred)[1], nrow(x))
  expect_equal(dim(pred)[2], 3)
  expect_true(all(pred[, 1] <= pred[, 3]))

  cors <- cor(y, pred)
  expect_true(cors[2] > cors[1])
  expect_true(cors[2] > cors[3])
  expect_true(cors[2] > 0.85)
})

test_that("Can use multi-output labels with built-in objectives", {
  data("mtcars")
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  y_mirrored <- cbind(y, -y)
  dm <- xgb.DMatrix(x, label = y_mirrored, nthread = n_threads)
  model <- xgb.train(
    params = xgb.params(
      tree_method = "hist",
      multi_strategy = "multi_output_tree",
      objective = "reg:squarederror",
      nthread = n_threads
    ),
    data = dm,
    nrounds = 5
  )
  pred <- predict(model, x)
  expect_equal(pred[, 1], -pred[, 2])
  expect_true(cor(y, pred[, 1]) > 0.9)
  expect_true(cor(y, pred[, 2]) < -0.9)
})

test_that("Can use multi-output labels with custom objectives", {
  data("mtcars")
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  y_mirrored <- cbind(y, -y)
  dm <- xgb.DMatrix(x, label = y_mirrored, nthread = n_threads)
  model <- xgb.train(
    params = xgb.params(
      tree_method = "hist",
      multi_strategy = "multi_output_tree",
      base_score = 0,
      objective = function(pred, dtrain) {
        y <- getinfo(dtrain, "label")
        grad <- pred - y
        hess <- rep(1, nrow(grad) * ncol(grad))
        hess <- matrix(hess, nrow = nrow(grad))
        return(list(grad = grad, hess = hess))
      },
      nthread = n_threads
    ),
    data = dm,
    nrounds = 5
  )
  pred <- predict(model, x)
  expect_equal(pred[, 1], -pred[, 2])
  expect_true(cor(y, pred[, 1]) > 0.9)
  expect_true(cor(y, pred[, 2]) < -0.9)
})

test_that("Can use ranking objectives with either 'qid' or 'group'", {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), nrow = 100)
  y <- sample(2, size = 100, replace = TRUE) - 1
  qid <- c(rep(1, 20), rep(2, 20), rep(3, 60))
  gr <- c(20, 20, 60)

  dmat_qid <- xgb.DMatrix(x, label = y, qid = qid)
  dmat_gr <- xgb.DMatrix(x, label = y, group = gr)

  params <- xgb.params(
    tree_method = "hist",
    lambdarank_num_pair_per_sample = 8,
    objective = "rank:ndcg",
    lambdarank_pair_method = "topk",
    nthread = n_threads
  )
  set.seed(123)
  model_qid <- xgb.train(params, dmat_qid, nrounds = 5)
  set.seed(123)
  model_gr <- xgb.train(params, dmat_gr, nrounds = 5)

  pred_qid <- predict(model_qid, x)
  pred_gr <- predict(model_gr, x)
  expect_equal(pred_qid, pred_gr)
})

test_that("Can predict on data.frame objects", {
  data("mtcars")
  y <- mtcars$mpg
  x_df <- mtcars[, -1]
  x_mat <- as.matrix(x_df)
  dm <- xgb.DMatrix(x_mat, label = y, nthread = n_threads)
  model <- xgb.train(
    params = xgb.params(
      tree_method = "hist",
      objective = "reg:squarederror",
      nthread = n_threads
    ),
    data = dm,
    nrounds = 5
  )

  pred_mat <- predict(model, xgb.DMatrix(x_mat))
  pred_df <- predict(model, x_df)
  expect_equal(pred_mat, unname(pred_df))
})

test_that("'base_margin' gives the same result in DMatrix as in inplace_predict", {
  data("mtcars")
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = y, nthread = n_threads)
  model <- xgb.train(
    params = xgb.params(
      tree_method = "hist",
      objective = "reg:squarederror",
      nthread = n_threads
    ),
    data = dm,
    nrounds = 5
  )

  set.seed(123)
  base_margin <- rnorm(nrow(x))
  dm_w_base <- xgb.DMatrix(data = x, base_margin = base_margin)
  pred_from_dm <- predict(model, dm_w_base)
  pred_from_mat <- predict(model, x, base_margin = base_margin)

  expect_equal(pred_from_dm, unname(pred_from_mat))
})

test_that("Coefficients from gblinear have the expected shape and names", {
  # Single-column coefficients
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  mm <- model.matrix(~., data = mtcars[, -1])
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  model <- xgb.train(
    data = dm,
    params = xgb.params(
      booster = "gblinear",
      nthread = 1
    ),
    nrounds = 3
  )
  coefs <- coef(model)
  expect_equal(length(coefs), ncol(x) + 1)
  expect_equal(names(coefs), c("(Intercept)", colnames(x)))
  pred_auto <- predict(model, x)
  pred_manual <- as.numeric(mm %*% coefs)
  expect_equal(pred_manual, unname(pred_auto), tolerance = 1e-5)

  # Multi-column coefficients
  data(iris)
  y <- as.numeric(iris$Species) - 1
  x <- as.matrix(iris[, -5])
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  mm <- model.matrix(~., data = iris[, -5])
  model <- xgb.train(
    data = dm,
    params = xgb.params(
      booster = "gblinear",
      objective = "multi:softprob",
      num_class = 3,
      nthread = 1
    ),
    nrounds = 3
  )
  coefs <- coef(model)
  expect_equal(nrow(coefs), ncol(x) + 1)
  expect_equal(ncol(coefs), 3)
  expect_equal(row.names(coefs), c("(Intercept)", colnames(x)))
  pred_auto <- predict(model, x, outputmargin = TRUE)
  pred_manual <- unname(mm %*% coefs)
  expect_equal(pred_manual, pred_auto, tolerance = 1e-7)

  # xgboost() with additional metadata
  model <- xgboost(
    iris[, -5],
    iris$Species,
    booster = "gblinear",
    objective = "multi:softprob",
    nrounds = 3,
    nthread = 1
  )
  coefs <- coef(model)
  expect_equal(row.names(coefs), c("(Intercept)", colnames(x)))
  expect_equal(colnames(coefs), levels(iris$Species))
})

test_that("Deep copies work as expected", {
  data(mtcars)
  y <- mtcars$mpg
  x <- mtcars[, -1]
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  model <- xgb.train(
   data = dm,
   params = xgb.params(nthread = 1),
   nrounds = 3
  )

  xgb.attr(model, "my_attr") <- 100
  model_shallow_copy <- model
  xgb.attr(model_shallow_copy, "my_attr") <- 333
  attr_orig <- xgb.attr(model, "my_attr")
  attr_shallow <- xgb.attr(model_shallow_copy, "my_attr")
  expect_equal(attr_orig, attr_shallow)

  model_deep_copy <- xgb.copy.Booster(model)
  xgb.attr(model_deep_copy, "my_attr") <- 444
  attr_orig <- xgb.attr(model, "my_attr")
  attr_deep <- xgb.attr(model_deep_copy, "my_attr")
  expect_false(attr_orig == attr_deep)
})

test_that("Pointer comparison works as expected", {
  library(xgboost)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  model <- xgb.train(
    params = xgb.params(nthread = 1),
    data = xgb.DMatrix(x, label = y, nthread = 1),
    nrounds = 3
  )

  model_shallow_copy <- model
  expect_true(xgb.is.same.Booster(model, model_shallow_copy))

  model_deep_copy <- xgb.copy.Booster(model)
  expect_false(xgb.is.same.Booster(model, model_deep_copy))

  xgb.attr(model_shallow_copy, "my_attr") <- 111
  expect_equal(xgb.attr(model, "my_attr"), "111")
  expect_null(xgb.attr(model_deep_copy, "my_attr"))
})

test_that("DMatrix field are set to booster when training", {
  set.seed(123)
  y <- rnorm(100)
  x <- matrix(rnorm(100 * 3), nrow = 100)
  x[, 2] <- abs(as.integer(x[, 2]))

  dm_unnamed <- xgb.DMatrix(x, label = y, nthread = 1)
  dm_feature_names <- xgb.DMatrix(x, label = y, feature_names = c("a", "b", "c"), nthread = 1)
  dm_feature_types <- xgb.DMatrix(x, label = y)
  setinfo(dm_feature_types, "feature_type", c("q", "c", "q"))
  dm_both <- xgb.DMatrix(x, label = y, feature_names = c("a", "b", "c"), nthread = 1)
  setinfo(dm_both, "feature_type", c("q", "c", "q"))

  params <- xgb.params(nthread = 1)
  model_unnamed <- xgb.train(data = dm_unnamed, params = params, nrounds = 3)
  model_feature_names <- xgb.train(data = dm_feature_names, params = params, nrounds = 3)
  model_feature_types <- xgb.train(data = dm_feature_types, params = params, nrounds = 3)
  model_both <- xgb.train(data = dm_both, params = params, nrounds = 3)

  expect_null(getinfo(model_unnamed, "feature_name"))
  expect_equal(getinfo(model_feature_names, "feature_name"), c("a", "b", "c"))
  expect_null(getinfo(model_feature_types, "feature_name"))
  expect_equal(getinfo(model_both, "feature_name"), c("a", "b", "c"))

  expect_null(variable.names(model_unnamed))
  expect_equal(variable.names(model_feature_names), c("a", "b", "c"))
  expect_null(variable.names(model_feature_types))
  expect_equal(variable.names(model_both), c("a", "b", "c"))

  expect_null(getinfo(model_unnamed, "feature_type"))
  expect_null(getinfo(model_feature_names, "feature_type"))
  expect_equal(getinfo(model_feature_types, "feature_type"), c("q", "c", "q"))
  expect_equal(getinfo(model_both, "feature_type"), c("q", "c", "q"))
})

test_that("Seed in params override PRNG from R", {
  set.seed(123)
  model1 <- xgb.train(
    data = xgb.DMatrix(
      agaricus.train$data,
      label = agaricus.train$label, nthread = 1L
    ),
    params = xgb.params(
      objective = "binary:logistic",
      max_depth = 3L,
      subsample = 0.1,
      colsample_bytree = 0.1,
      seed = 111L
    ),
    nrounds = 3L
  )

  set.seed(456)
  model2 <- xgb.train(
    data = xgb.DMatrix(
      agaricus.train$data,
      label = agaricus.train$label, nthread = 1L
    ),
    params = xgb.params(
      objective = "binary:logistic",
      max_depth = 3L,
      subsample = 0.1,
      colsample_bytree = 0.1,
      seed = 111L
    ),
    nrounds = 3L
  )

  expect_equal(
    xgb.save.raw(model1, raw_format = "json"),
    xgb.save.raw(model2, raw_format = "json")
  )

  set.seed(123)
  model3 <- xgb.train(
    data = xgb.DMatrix(
      agaricus.train$data,
      label = agaricus.train$label, nthread = 1L
    ),
    params = xgb.params(
      objective = "binary:logistic",
      max_depth = 3L,
      subsample = 0.1,
      colsample_bytree = 0.1,
      seed = 222L
    ),
    nrounds = 3L
  )
  expect_false(
    isTRUE(
      all.equal(
        xgb.save.raw(model1, raw_format = "json"),
        xgb.save.raw(model3, raw_format = "json")
      )
    )
  )
})

test_that("xgb.cv works for AFT", {
  X <- matrix(c(1, -1, -1, 1, 0, 1, 1, 0), nrow = 4, byrow = TRUE)  # 4x2 matrix
  dtrain <- xgb.DMatrix(X, nthread = n_threads)

  params <- xgb.params(objective = 'survival:aft', learning_rate = 0.2, max_depth = 2L, nthread = n_threads)

  # data must have bounds
  expect_error(
    xgb.cv(
      params = params,
      data = dtrain,
      nround = 5L,
      nfold = 4L
    )
  )

  setinfo(dtrain, 'label_lower_bound', c(2, 3, 0, 4))
  setinfo(dtrain, 'label_upper_bound', c(2, Inf, 4, 5))

  # automatic stratified splitting is turned off
  expect_warning(
    xgb.cv(
      params = params, data = dtrain, nround = 5L, nfold = 4L,
      stratified = TRUE, verbose = FALSE
    )
  )

  # this works without any issue
  expect_no_warning(
    xgb.cv(params = params, data = dtrain, nround = 5L, nfold = 4L, verbose = FALSE)
  )
})

test_that("xgb.cv works for ranking", {
  data(iris)
  x <- iris[, -(4:5)]
  y <- as.integer(iris$Petal.Width)
  group <- rep(50, 3)
  dm <- xgb.DMatrix(x, label = y, group = group)
  res <- xgb.cv(
    data = dm,
    params = xgb.params(
      objective = "rank:pairwise",
      max_depth = 3,
      nthread = 1L
    ),
    nrounds = 3,
    nfold = 2,
    verbose = FALSE,
    stratified = FALSE
  )
  expect_equal(length(res$folds), 2L)
})

test_that("Row names are preserved in outputs", {
  data(iris)
  x <- iris[, -5]
  y <- as.numeric(iris$Species) - 1
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  model <- xgb.train(
    data = dm,
    params = xgb.params(
      objective = "multi:softprob",
      num_class = 3,
      max_depth = 2,
      nthread = 1
    ),
    nrounds = 3
  )
  row.names(x) <- paste0("r", seq(1, nrow(x)))
  pred <- predict(model, x)
  expect_equal(row.names(pred), row.names(x))
  pred <- predict(model, x, avoid_transpose = TRUE)
  expect_equal(colnames(pred), row.names(x))

  data(mtcars)
  y <- mtcars[, 1]
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(data = x, label = y)
  model <- xgb.train(
    data = dm,
    params = xgb.params(
      max_depth = 2,
      nthread = 1
    ),
    nrounds = 3
  )
  row.names(x) <- paste0("r", seq(1, nrow(x)))
  pred <- predict(model, x)
  expect_equal(names(pred), row.names(x))
  pred <- predict(model, x, avoid_transpose = TRUE)
  expect_equal(names(pred), row.names(x))
  pred <- predict(model, x, predleaf = TRUE)
  expect_equal(row.names(pred), row.names(x))
  pred <- predict(model, x, predleaf = TRUE, avoid_transpose = TRUE)
  expect_equal(colnames(pred), row.names(x))
})
