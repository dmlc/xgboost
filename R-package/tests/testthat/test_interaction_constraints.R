require(xgboost)

context("interaction constraints")

n_threads <- 2

set.seed(1024)
x1 <- rnorm(1000, 1)
x2 <- rnorm(1000, 1)
x3 <- sample(c(1, 2, 3), size = 1000, replace = TRUE)
y <- x1 + x2 + x3 + x1 * x2 * x3 + rnorm(1000, 0.001) + 3 * sin(x1)
train <- matrix(c(x1, x2, x3), ncol = 3)

test_that("interaction constraints scientific representation", {
  rows <- 10
  ## When number exceeds 1e5, R paste function uses scientific representation.
  ## See: https://github.com/dmlc/xgboost/issues/5179
  cols <- 1e5 + 10

  d <- matrix(rexp(rows, rate = .1), nrow = rows, ncol = cols)
  y <- rnorm(rows)

  dtrain <- xgb.DMatrix(data = d, info = list(label = y), nthread = n_threads)
  inc <- list(c(seq.int(from = 0, to = cols, by = 1)))

  with_inc <- xgb.train(
    data = dtrain,
    tree_method = 'hist',
    interaction_constraints = inc,
    nrounds = 10,
    nthread = n_threads
  )
  without_inc <- xgb.train(
    data = dtrain, tree_method = 'hist', nrounds = 10, nthread = n_threads
  )
  expect_equal(xgb.save.raw(with_inc), xgb.save.raw(without_inc))
})
