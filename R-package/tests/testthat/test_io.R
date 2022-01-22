context("Test model IO.")
## some other tests are in test_basic.R
require(xgboost)
require(testthat)

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

  from_json <- xgb.load.raw(json_bytes)
  from_ubj <- xgb.load.raw(ubj_bytes)

  ## FIXME(jiamingy): Should we include these 3 lines into `xgb.load.raw`?
  from_json <- list(handle = from_json, raw = NULL)
  class(from_json) <- "xgb.Booster"
  from_json <- xgb.Booster.complete(from_json, saveraw = TRUE)

  from_ubj <- list(handle = from_ubj, raw = NULL)
  class(from_ubj) <- "xgb.Booster"
  from_ubj <- xgb.Booster.complete(from_ubj, saveraw = TRUE)

  json2old <- xgb.save.raw(from_json, raw_format = "deprecated")
  ubj2old <- xgb.save.raw(from_ubj, raw_format = "deprecated")

  expect_equal(json2old, ubj2old)
  expect_equal(json2old, old_bytes)
})
