context("Test model IO.")

data(agaricus.train, package = "xgboost")
data(agaricus.test, package = "xgboost")
train <- agaricus.train
test <- agaricus.test

test_that("load/save raw works", {
  nrounds <- 8
  booster <- xgb.train(
    data = xgb.DMatrix(train$data, label = train$label),
    nrounds = nrounds,
    params = xgb.params(
      objective = "binary:logistic",
      nthread = 2
    )
  )

  json_bytes <- xgb.save.raw(booster, raw_format = "json")
  ubj_bytes <- xgb.save.raw(booster, raw_format = "ubj")
  old_bytes <- xgb.save.raw(booster, raw_format = "deprecated")

  from_json <- xgb.load.raw(json_bytes)
  from_ubj <- xgb.load.raw(ubj_bytes)

  json2old <- xgb.save.raw(from_json, raw_format = "deprecated")
  ubj2old <- xgb.save.raw(from_ubj, raw_format = "deprecated")

  expect_equal(json2old, ubj2old)
  expect_equal(json2old, old_bytes)
})

test_that("saveRDS preserves C and R attributes", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  model <- xgb.train(
    data = dm,
    params = xgb.params(nthread = 1, max_depth = 2),
    nrounds = 5
  )
  attributes(model)$my_attr <- "qwerty"
  xgb.attr(model, "c_attr") <- "asdf"

  fname <- file.path(tempdir(), "xgb_model.Rds")
  saveRDS(model, fname)
  model_new <- readRDS(fname)

  expect_equal(attributes(model_new)$my_attr, attributes(model)$my_attr)
  expect_equal(xgb.attr(model, "c_attr"), xgb.attr(model_new, "c_attr"))
})

test_that("R serializers keep C config", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  model <- xgb.train(
    data = dm,
    params = list(
      tree_method = "approx",
      nthread = 1,
      max_depth = 2
    ),
    nrounds = 3
  )
  model_new <- unserialize(serialize(model, NULL))
  expect_equal(
    xgb.config(model)$learner$gradient_booster$gbtree_train_param$tree_method,
    xgb.config(model_new)$learner$gradient_booster$gbtree_train_param$tree_method
  )
  expect_equal(variable.names(model), variable.names(model_new))
})
