context("Test model IO.")

data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")
train <- agaricus.train
test <- agaricus.test

test_that("load/save raw works", {
  nrounds <- 8
  booster <- xgboost(
    data = train$data, label = train$label,
    nrounds = nrounds, objective = "binary:logistic"
  )

  json_bytes <- xgb.save.raw(booster, raw_format = "json")
  ubj_bytes <- xgb.save.raw(booster, raw_format = "ubj")
  old_bytes <- xgb.save.raw(booster, raw_format = "deprecated")

  from_json <- xgb.load.raw(json_bytes, as_booster = TRUE)
  from_ubj <- xgb.load.raw(ubj_bytes, as_booster = TRUE)

  json2old <- xgb.save.raw(from_json, raw_format = "deprecated")
  ubj2old <- xgb.save.raw(from_ubj, raw_format = "deprecated")

  expect_equal(json2old, ubj2old)
  expect_equal(json2old, old_bytes)
})
