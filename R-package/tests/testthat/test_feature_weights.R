context("feature weights")

test_that("training with feature weights works", {
  nrows <- 1000
  ncols <- 9
  set.seed(2022)
  x <- matrix(rnorm(nrows * ncols), nrow = nrows)
  y <- rowSums(x)
  weights <- seq(from = 1, to = ncols)

  test <- function(tm) {
    names <- paste0("f", 1:ncols)
    xy <- xgb.DMatrix(data = x, label = y, feature_weights = weights)
    params <- list(colsample_bynode = 0.4, tree_method = tm, nthread = 1)
    model <- xgb.train(params = params, data = xy, nrounds = 32)
    importance <- xgb.importance(model = model, feature_names = names)
    expect_equal(dim(importance), c(ncols, 4))
    importance <- importance[order(importance$Feature)]
    expect_lt(importance[1, Frequency], importance[9, Frequency])
  }

  for (tm in c("hist", "approx", "exact")) {
    test(tm)
  }
})
