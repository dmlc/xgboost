context("Test Unicode handling")

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
train <- agaricus.train
test <- agaricus.test
set.seed(1994)

test_that("Can save and load models with Unicode paths", {
  nrounds <- 2
  bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
                 eta = 1, nthread = 2, nrounds = nrounds, objective = "binary:logistic",
                 eval_metric = "error")
  tmpdir <- tempdir()
  lapply(c("모델.json", "がうる・ぐら.json", "类继承.ubj"), function(x) {
    path <- file.path(tmpdir, x)
    xgb.save(bst, path)
    bst2 <- xgb.load(path)
    expect_equal(predict(bst, test$data), predict(bst2, test$data))
  })
})
