context('Test models with custom objective')

set.seed(1994)

n_threads <- 2

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
dtrain <- xgb.DMatrix(
  agaricus.train$data, label = agaricus.train$label, nthread = n_threads
)
dtest <- xgb.DMatrix(
  agaricus.test$data, label = agaricus.test$label, nthread = n_threads
)
evals <- list(eval = dtest, train = dtrain)

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

param <- list(max_depth = 2, learning_rate = 1, nthread = n_threads,
              objective = logregobj, eval_metric = evalerror)
num_round <- 2

test_that("custom objective works", {
  bst <- xgb.train(param, dtrain, num_round, evals, verbose = 0)
  expect_equal(class(bst), "xgb.Booster")
  expect_false(is.null(attributes(bst)$evaluation_log))
  expect_false(is.null(attributes(bst)$evaluation_log$eval_error))
  expect_lt(attributes(bst)$evaluation_log[num_round, eval_error], 0.03)
})

test_that("custom objective in CV works", {
  cv <- xgb.cv(param, dtrain, num_round, nfold = 10, verbose = FALSE, stratified = FALSE)
  expect_false(is.null(cv$evaluation_log))
  expect_equal(dim(cv$evaluation_log), c(2, 5))
  expect_lt(cv$evaluation_log[num_round, test_error_mean], 0.03)
})

test_that("custom objective with early stop works", {
  bst <- xgb.train(param, dtrain, 10, evals, verbose = 0)
  expect_equal(class(bst), "xgb.Booster")
  train_log <- attributes(bst)$evaluation_log$train_error
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
  bst <- xgb.train(param, dtrain, num_round, evals, verbose = 0)
  expect_equal(class(bst), "xgb.Booster")
})

test_that("custom objective with multi-class shape", {
  data <- as.matrix(iris[, -5])
  label <-  as.numeric(iris$Species) - 1
  dtrain <- xgb.DMatrix(data = data, label = label, nthread = n_threads)
  n_classes <- 3

  fake_softprob <- function(preds, dtrain) {
    expect_true(all(matrix(preds) == 0.5))
    ## use numeric vector here to test compatibility with XGBoost < 2.1
    grad <- rnorm(length(as.matrix(preds)))
    expect_equal(dim(data)[1] * n_classes, dim(as.matrix(preds))[1] * n_classes)
    hess <- rnorm(length(as.matrix(preds)))
    return(list(grad = grad, hess = hess))
  }
  fake_merror <- function(preds, dtrain) {
    expect_equal(dim(data)[1] * n_classes, dim(as.matrix(preds))[1])
  }
  param$objective <- fake_softprob
  param$eval_metric <- fake_merror
  expect_warning({
    bst <- xgb.train(c(param, list(num_class = n_classes)), dtrain, nrounds = 1)
  })
})

softmax <- function(values) {
  values <- as.numeric(values)
  exps <- exp(values)
  den <- sum(exps)
  return(exps / den)
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
      g <- if (c - 1 == t) {
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
  data <- as.matrix(iris[, -5])
  label <- as.numeric(iris$Species) - 1

  dtrain <- xgb.DMatrix(data = data, label = label)

  param$num_class <- 3
  param$objective <- softprob
  param$eval_metric <- "merror"
  param$base_score <- 0.5

  custom_bst <- xgb.train(param, dtrain, 2)
  custom_predt <- predict(custom_bst, dtrain)

  param$objective <- "multi:softmax"
  builtin_bst <- xgb.train(param, dtrain, 2)
  builtin_predt <- predict(builtin_bst, dtrain)

  expect_equal(custom_predt, builtin_predt)
})

test_that("custom metric with multi-target passes reshaped data to feval", {
  x <- as.matrix(iris[, -5])
  y <- as.numeric(iris$Species) - 1
  dtrain <- xgb.DMatrix(data = x, label = y)

  multinomial.ll <- function(predt, dtrain) {
    expect_equal(dim(predt), c(nrow(iris), 3L))
    y <- getinfo(dtrain, "label")
    probs <- apply(predt, 1, softmax) |> t()
    probs.y <- probs[cbind(seq(1L, nrow(predt)), y + 1L)]
    ll <- sum(log(probs.y))
    return(list(metric = "multinomial-ll", value = -ll))
  }

  model <- xgb.train(
    params = list(
      objective = "multi:softmax",
      num_class = 3L,
      base_score = 0,
      disable_default_eval_metric = TRUE,
      eval_metric = multinomial.ll,
      max_depth = 123,
      seed = 123
    ),
    data = dtrain,
    nrounds = 2L,
    evals = list(Train = dtrain),
    verbose = 0
  )

  model <- xgb.train(
    params = list(
      objective = "multi:softmax",
      num_class = 3L,
      base_score = 0,
      disable_default_eval_metric = TRUE,
      max_depth = 123,
      seed = 123
    ),
    data = dtrain,
    nrounds = 2L,
    evals = list(Train = dtrain),
    custom_metric = multinomial.ll,
    verbose = 0
  )
})
