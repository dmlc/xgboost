require(xgboost)

context("basic functions")

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train = agaricus.train
test = agaricus.test

test_that("train and predict", {
  bst = xgboost(data = train$data, label = train$label, max.depth = 2,
                eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
  pred = predict(bst, test$data)
})


test_that("early stopping", {
  res = xgb.cv(data = train$data, label = train$label, max.depth = 2, nfold = 5,
               eta = 0.3, nthread = 2, nround = 20, objective = "binary:logistic",
               early.stop.round = 3, maximize = FALSE)
  expect_true(nrow(res)<20)
  bst = xgboost(data = train$data, label = train$label, max.depth = 2,
                eta = 0.3, nthread = 2, nround = 20, objective = "binary:logistic",
                early.stop.round = 3, maximize = FALSE)
  pred = predict(bst, test$data)
})

test_that("save_period", {
  bst = xgboost(data = train$data, label = train$label, max.depth = 2,
                eta = 0.3, nthread = 2, nround = 20, objective = "binary:logistic",
                save_period = 10, save_name = "xgb.model")
  pred = predict(bst, test$data)
})
