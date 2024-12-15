context("Test Unicode handling")

data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
train <- agaricus.train
test <- agaricus.test
set.seed(1994)

test_that("Can save and load models with Unicode paths", {
  nrounds <- 2
  bst <- xgb.train(
    data = xgb.DMatrix(train$data, label = train$label),
    nrounds = nrounds,
    params = xgb.params(
      max_depth = 2,
      nthread = 2,
      objective = "binary:logistic"
    )
  )
  tmpdir <- tempdir()
  lapply(c("모델.json", "がうる・ぐら.json", "类继承.ubj"), function(x) {
    path <- file.path(tmpdir, x)
    xgb.save(bst, path)
    bst2 <- xgb.load(path)
    xgb.model.parameters(bst2) <- list(nthread = 2)
    expect_equal(predict(bst, test$data), predict(bst2, test$data))
  })
})
