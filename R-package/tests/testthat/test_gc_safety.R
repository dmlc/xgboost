require(xgboost)

context("Garbage Collection Safety Check")

test_that("train and prediction when gctorture is on", {
  data(agaricus.train, package='xgboost')
  data(agaricus.test, package='xgboost')
  train <- agaricus.train
  test <- agaricus.test
  gctorture(TRUE)
  bst <- xgboost(data = train$data, label = train$label, max.depth = 2,
  eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
  pred <- predict(bst, test$data)
  gctorture(FALSE)
})
