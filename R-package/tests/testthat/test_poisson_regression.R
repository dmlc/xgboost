context('Test Poisson regression model')

set.seed(1994)

test_that("Poisson regression works", {
  data(mtcars)
  bst <- xgb.train(
    data = xgb.DMatrix(as.matrix(mtcars[, -11]), label = mtcars[, 11]),
    nrounds = 10, verbose = 0,
    params = xgb.params(objective = 'count:poisson',  nthread = 2)
  )
  expect_equal(class(bst), "xgb.Booster")
  pred <- predict(bst, as.matrix(mtcars[, -11]))
  expect_equal(length(pred), 32)
  expect_lt(sqrt(mean((pred - mtcars[, 11])^2)), 1.2)
})

test_that("Poisson regression is centered around mean", {
  m <- 50L
  n <- 10L
  y <- rpois(m, n)
  x <- matrix(rnorm(m * n), nrow = m)
  model <- xgb.train(
    data = xgb.DMatrix(x, label = y),
    params = xgb.params(objective = "count:poisson", min_split_loss = 1e4),
    nrounds = 1
  )
  model_json <- xgb.save.raw(model, "json") |> rawToChar() |> jsonlite::fromJSON()
  expect_equal(
    model_json$learner$learner_model_param$base_score |> as.numeric(),
    mean(y),
    tolerance = 1e-4
  )

  pred <- predict(model, x)
  expect_equal(
    pred,
    rep(mean(y), m),
    tolerance = 1e-4
  )

  w <- y + 1
  model_weighted <- xgb.train(
    data = xgb.DMatrix(x, label = y, weight = w),
    params = xgb.params(objective = "count:poisson", min_split_loss = 1e4),
    nrounds = 1
  )
  model_json <- xgb.save.raw(model_weighted, "json") |> rawToChar() |> jsonlite::fromJSON()
  expect_equal(
    model_json$learner$learner_model_param$base_score |> as.numeric(),
    weighted.mean(y, w),
    tolerance = 1e-4
  )
})
