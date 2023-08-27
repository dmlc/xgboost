context('Test models with custom objective')

set.seed(1994)

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)
watchlist <- list(eval = dtest, train = dtrain)

logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1 / (1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0.5))) / length(labels)
  return(list(metric = "error", value = err))
}

param <- list(max_depth = 2, eta = 1, nthread = 2,
              objective = logregobj, eval_metric = evalerror)
num_round <- 2

test_that("custom objective works", {
  bst <- xgb.train(param, dtrain, num_round, watchlist)
  expect_equal(class(bst), "xgb.Booster")
  expect_false(is.null(bst$evaluation_log))
  expect_false(is.null(bst$evaluation_log$eval_error))
  expect_lt(bst$evaluation_log[num_round, eval_error], 0.03)
})

test_that("custom objective in CV works", {
  cv <- xgb.cv(param, dtrain, num_round, nfold = 10, verbose = FALSE)
  expect_false(is.null(cv$evaluation_log))
  expect_equal(dim(cv$evaluation_log), c(2, 5))
  expect_lt(cv$evaluation_log[num_round, test_error_mean], 0.03)
})

test_that("custom objective with early stop works", {
  bst <- xgb.train(param, dtrain, 10, watchlist)
  expect_equal(class(bst), "xgb.Booster")
  train_log <- bst$evaluation_log$train_error
  expect_true(all(diff(train_log) <= 0))
})

test_that("custom objective using DMatrix attr works", {

  attr(dtrain, 'label') <- getinfo(dtrain, 'label')

  logregobjattr <- function(preds, dtrain) {
    labels <- attr(dtrain, 'label')
    preds <- 1 / (1 + exp(-preds))
    grad <- preds - labels
    hess <- preds * (1 - preds)
    return(list(grad = grad, hess = hess))
  }
  param$objective <- logregobjattr
  bst <- xgb.train(param, dtrain, num_round, watchlist)
  expect_equal(class(bst), "xgb.Booster")
})

test_that("custom objective with multi-class shape", {
  data <- as.matrix(iris[, -5])
  label <-  as.numeric(iris$Species) - 1
  dtrain <- xgb.DMatrix(data = data, label = label)
  nclasses <- 3

  fake_softprob <- function(preds, dtrain) {
    expect_true(all(matrix(preds) == 0.5))
    grad <- rnorm(dim(as.matrix(preds))[1])
    expect_equal(dim(data)[1] * nclasses, dim(as.matrix(preds))[1])
    hess <- rnorm(dim(as.matrix(preds))[1])
    return (list(grad = grad, hess = hess))
  }
  fake_merror <- function(preds, dtrain) {
    expect_equal(dim(data)[1] * nclasses, dim(as.matrix(preds))[1])
  }
  param$objective <- fake_softprob
  param$eval_metric <- fake_merror
  bst <- xgb.train(param, dtrain, 1, num_class = nclasses)
})

softmax <- function(proba) {
  proba <- as.numeric(proba)
  exps <- exp(proba)
  den <- sum(exps)
  return (exps / den)
}

softprob <- function(predt, dtrain) {
  y <- getinfo(dtrain, "label")

  n_samples <- dim(predt)[1]
  n_classes <- dim(predt)[2]

  grad <- matrix(nrow = n_samples, ncol = n_classes)
  hess <- matrix(nrow = n_samples, ncol = n_classes)

  for (i in seq_len(n_samples)) {
    t <- y[i]
    p <- softmax(predt[i, ])
    for (c in seq_len(n_classes)) {
      g <- if (c == t) {
        p[c] - 1.0
      } else {
        p[c]
      }
      h <- max((2.0 * p[c] * (1.0 - p[c])), 1e-6)
      grad[i, c] <- g
      hess[i, c] <- h
    }
  }

  return(list(grad = grad, hess = hess))
}

test_that("custom objective with multi-class works", {
  param$objective <- softprob
  param$eval_metric <- "merror"
  custom_bst <- xgb.train(param, dtrain, 4)

  param$objective <- "multi:softmax"
  builtin_bst <- xgb.train(param, dtrain, 4)
})
