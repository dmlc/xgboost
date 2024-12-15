context("testing xgb.Booster slicing")

data(agaricus.train, package = "xgboost")
dm <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label, nthread = 1)
# Note: here need large step sizes in order for the predictions
# to have substantially different leaf assignments on each tree
model <- xgb.train(
  params = xgb.params(objective = "binary:logistic", nthread = 1, max_depth = 4),
  data = dm,
  nrounds = 20
)
pred <- predict(model, dm, predleaf = TRUE)

test_that("Slicing full model", {
  new_model <- xgb.slice.Booster(model, 1, 0)
  expect_equal(xgb.save.raw(new_model), xgb.save.raw(model))

  new_model <- model[]
  expect_equal(xgb.save.raw(new_model), xgb.save.raw(model))

  new_model <- model[1:length(model)] # nolint
  expect_equal(xgb.save.raw(new_model), xgb.save.raw(model))
})

test_that("Slicing sequence from start", {
  new_model <- xgb.slice.Booster(model, 1, 10)
  new_pred <- predict(new_model, dm, predleaf = TRUE)
  expect_equal(new_pred, pred[, seq(1, 10)])

  new_model <- model[1:10]
  new_pred <- predict(new_model, dm, predleaf = TRUE)
  expect_equal(new_pred, pred[, seq(1, 10)])
})

test_that("Slicing sequence from middle", {
  new_model <- xgb.slice.Booster(model, 5, 10)
  new_pred <- predict(new_model, dm, predleaf = TRUE)
  expect_equal(new_pred, pred[, seq(5, 10)])

  new_model <- model[5:10]
  new_pred <- predict(new_model, dm, predleaf = TRUE)
  expect_equal(new_pred, pred[, seq(5, 10)])
})

test_that("Slicing with non-unit step", {
  for (s in 2:5) {
    new_model <- xgb.slice.Booster(model, 1, 17, s)
    new_pred <- predict(new_model, dm, predleaf = TRUE)
    expect_equal(new_pred, pred[, seq(1, 17, s)])

    new_model <- model[seq(1, 17, s)]
    new_pred <- predict(new_model, dm, predleaf = TRUE)
    expect_equal(new_pred, pred[, seq(1, 17, s)])
  }
})

test_that("Slicing with non-unit step from middle", {
  for (s in 2:5) {
    new_model <- xgb.slice.Booster(model, 4, 17, s)
    new_pred <- predict(new_model, dm, predleaf = TRUE)
    expect_equal(new_pred, pred[, seq(4, 17, s)])

    new_model <- model[seq(4, 17, s)]
    new_pred <- predict(new_model, dm, predleaf = TRUE)
    expect_equal(new_pred, pred[, seq(4, 17, s)])
  }
})
