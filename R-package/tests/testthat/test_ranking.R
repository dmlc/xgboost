context('Learning to rank')

n_threads <- 2

test_that('Test ranking with unweighted data', {
  X <- Matrix::sparseMatrix(
    i = c(2, 3, 7, 9, 12, 15, 17, 18)
    , j = c(1, 1, 2, 2,  3,  3,  4,  4)
    , x = rep(1.0, 8)
    , dims = c(20, 4)
  )
  y <- c(0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0)
  group <- c(5, 5, 5, 5)
  dtrain <- xgb.DMatrix(X, label = y, group = group, nthread = n_threads)

  params <- xgb.params(
    learning_rate = 1,
    tree_method = 'exact',
    objective = 'rank:pairwise',
    max_depth = 1,
    eval_metric = c('auc', 'aucpr'),
    nthread = n_threads
  )
  bst <- xgb.train(params, dtrain, nrounds = 10, evals = list(train = dtrain), verbose = 0)
  # Check if the metric is monotone increasing
  expect_true(all(diff(attributes(bst)$evaluation_log$train_auc) >= 0))
  expect_true(all(diff(attributes(bst)$evaluation_log$train_aucpr) >= 0))
})

test_that('Test ranking with weighted data', {
  X <- Matrix::sparseMatrix(
    i = c(2, 3, 7, 9, 12, 15, 17, 18)
    , j = c(1, 1, 2, 2,  3,  3,  4,  4)
    , x = rep(1.0, 8)
    , dims = c(20, 4)
  )
  y <- c(0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0)
  group <- c(5, 5, 5, 5)
  weight <- c(1.0, 2.0, 3.0, 4.0)
  dtrain <- xgb.DMatrix(
    X, label = y, group = group, weight = weight, nthread = n_threads
  )

  params <- xgb.params(
    learning_rate = 1,
    tree_method = "exact",
    objective = "rank:pairwise",
    max_depth = 1,
    eval_metric = c("auc", "aucpr"),
    nthread = n_threads
  )
  bst <- xgb.train(params, dtrain, nrounds = 10, evals = list(train = dtrain), verbose = 0)
  # Check if the metric is monotone increasing
  expect_true(all(diff(attributes(bst)$evaluation_log$train_auc) >= 0))
  expect_true(all(diff(attributes(bst)$evaluation_log$train_aucpr) >= 0))
  for (i in 1:10) {
    pred <- predict(bst, newdata = dtrain, iterationrange = c(1, i))
    # is_sorted[i]: is i-th group correctly sorted by the ranking predictor?
    is_sorted <- lapply(seq(1, 20, by = 5),
      function(k) {
        ind <- order(-pred[k:(k + 4)])
        z <- y[ind + (k - 1)]
        all(diff(z) <= 0)  # Check if z is monotone decreasing
      })
    # Since we give weights 1, 2, 3, 4 to the four query groups,
    # the ranking predictor will first try to correctly sort the last query group
    # before correctly sorting other groups.
    expect_true(all(diff(as.numeric(is_sorted)) >= 0))
  }
})
