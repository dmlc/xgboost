context('Learning to rank')

test_that('Test ranking with unweighted data', {
  X <- Matrix::sparseMatrix(
    i = c(2, 3, 7, 9, 12, 15, 17, 18)
    , j = c(1, 1, 2, 2,  3,  3,  4,  4)
    , x = rep(1.0, 8)
    , dims = c(20, 4)
  )
  y <- c(0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0)
  group <- c(5, 5, 5, 5)
  dtrain <- xgb.DMatrix(X, label = y, group = group)

  params <- list(eta = 1, tree_method = 'exact', objective = 'rank:pairwise', max_depth = 1,
                 eval_metric = 'auc', eval_metric = 'aucpr')
  bst <- xgb.train(params, dtrain, nrounds = 10, watchlist = list(train = dtrain))
  # Check if the metric is monotone increasing
  expect_true(all(diff(bst$evaluation_log$train_auc) >= 0))
  expect_true(all(diff(bst$evaluation_log$train_aucpr) >= 0))
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
  dtrain <- xgb.DMatrix(X, label = y, group = group, weight = weight)

  params <- list(eta = 1, tree_method = 'exact', objective = 'rank:pairwise', max_depth = 1,
                 eval_metric = 'auc', eval_metric = 'aucpr')
  bst <- xgb.train(params, dtrain, nrounds = 10, watchlist = list(train = dtrain))
  # Check if the metric is monotone increasing
  expect_true(all(diff(bst$evaluation_log$train_auc) >= 0))
  expect_true(all(diff(bst$evaluation_log$train_aucpr) >= 0))
  for (i in 1:10) {
    pred <- predict(bst, newdata = dtrain, ntreelimit = i)
    # is_sorted[i]: is i-th group correctly sorted by the ranking predictor?
    is_sorted <- lapply(seq(1, 20, by = 5),
      function (k) {
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
