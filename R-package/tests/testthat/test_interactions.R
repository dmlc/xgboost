context('Test prediction of feature interactions')

require(xgboost)

set.seed(123)
n_threads <- 2

test_that("SHAP contribution values are not NAN", {
  d <- data.frame(
    x1 = c(-2.3, 1.4, 5.9, 2, 2.5, 0.3, -3.6, -0.2, 0.5, -2.8, -4.6, 3.3, -1.2,
           -1.1, -2.3, 0.4, -1.5, -0.2, -1, 3.7),
    x2 = c(291.179171, 269.198331, 289.942097, 283.191669, 269.673332,
           294.158346, 287.255835, 291.530838, 285.899586, 269.290833,
           268.649586, 291.530841, 280.074593, 269.484168, 293.94042,
           294.327506, 296.20709, 295.441669, 283.16792, 270.227085),
    y = c(9, 15, 5.7, 9.2, 22.4, 5, 9, 3.2, 7.2, 13.1, 7.8, 16.9, 6.5, 22.1,
          5.3, 10.4, 11.1, 13.9, 11, 20.5),
    fold = c(2, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2))

  ivs <- c("x1", "x2")

  fit <- xgboost(
    verbose = 0,
    params = list(
      objective = "reg:squarederror",
      eval_metric = "rmse",
      nthread = n_threads
    ),
    data = as.matrix(subset(d, fold == 2)[, ivs]),
    label = subset(d, fold == 2)$y,
    nrounds = 3
  )

  shaps <- as.data.frame(predict(fit,
    newdata = as.matrix(subset(d, fold == 1)[, ivs]),
    predcontrib = TRUE))
  result <- cbind(shaps, sum = rowSums(shaps), pred = predict(fit,
      newdata = as.matrix(subset(d, fold == 1)[, ivs])))

  expect_true(identical(TRUE, all.equal(result$sum, result$pred, tol = 1e-6)))
})


test_that("multiclass feature interactions work", {
  dm <- xgb.DMatrix(
    as.matrix(iris[, -5]), label = as.numeric(iris$Species) - 1, nthread = n_threads
  )
  param <- list(
    eta = 0.1, max_depth = 4, objective = 'multi:softprob', num_class = 3, nthread = n_threads
  )
  b <- xgb.train(param, dm, 40)
  pred <- t(
    array(
      data = predict(b, dm, outputmargin = TRUE),
      dim = c(3, 150)
    )
  )

  # SHAP contributions:
  cont <- predict(b, dm, predcontrib = TRUE)
  expect_length(cont, 3)
  # rewrap them as a 3d array
  cont <- array(
    data = unlist(cont),
    dim = c(150, 5,  3)
  )

  # make sure for each row they add up to marginal predictions
  expect_lt(max(abs(apply(cont, c(1, 3), sum) - pred)), 0.001)

  # SHAP interaction contributions:
  intr <- predict(b, dm, predinteraction = TRUE)
  expect_length(intr, 3)
  # rewrap them as a 4d array
  intr <- aperm(
    a = array(
      data = unlist(intr),
      dim = c(150, 5, 5, 3)
    ),
    perm = c(4, 1, 2, 3)  # [grp, row, col, col]
  )

  # check the symmetry
  expect_lt(max(abs(aperm(intr, c(1, 2, 4, 3)) - intr)), 0.00001)
  # sums WRT columns must be close to feature contributions
  expect_lt(max(abs(apply(intr, c(1, 2, 3), sum) - aperm(cont, c(3, 1, 2)))), 0.00001)
})


test_that("SHAP single sample works", {
  train <- agaricus.train
  test <- agaricus.test
  booster <- xgboost(
    data = train$data,
    label = train$label,
    max_depth = 2,
    nrounds = 4,
    objective = "binary:logistic",
    nthread = n_threads
  )

  predt <- predict(
    booster,
    newdata = train$data[1, , drop = FALSE], predcontrib = TRUE
  )
  expect_equal(dim(predt), c(1, dim(train$data)[2] + 1))

  predt <- predict(
    booster,
    newdata = train$data[1, , drop = FALSE], predinteraction = TRUE
  )
  expect_equal(dim(predt), c(1, dim(train$data)[2] + 1, dim(train$data)[2] + 1))
})
