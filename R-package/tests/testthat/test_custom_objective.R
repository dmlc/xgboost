context('Test models with custom objective')

require(xgboost)

test_that("custom objective works", {
  data(agaricus.train, package='xgboost')
  data(agaricus.test, package='xgboost')
  dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
  dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)
  
  watchlist <- list(eval = dtest, train = dtrain)
  num_round <- 2
  
  logregobj <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    preds <- 1/(1 + exp(-preds))
    grad <- preds - labels
    hess <- preds * (1 - preds)
    return(list(grad = grad, hess = hess))
  }
  evalerror <- function(preds, dtrain) {
    labels <- getinfo(dtrain, "label")
    err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
    return(list(metric = "error", value = err))
  }
  
  param <- list(max.depth=2, eta=1, nthread = 2, silent=1, 
                objective=logregobj, eval_metric=evalerror)
  
  bst <- xgb.train(param, dtrain, num_round, watchlist)
  expect_equal(class(bst), "xgb.Booster")
  expect_equal(length(bst$raw), 1064)
  attr(dtrain, 'label') <- getinfo(dtrain, 'label')
  
  logregobjattr <- function(preds, dtrain) {
    labels <- attr(dtrain, 'label')
    preds <- 1/(1 + exp(-preds))
    grad <- preds - labels
    hess <- preds * (1 - preds)
    return(list(grad = grad, hess = hess))
  }
  param <- list(max.depth=2, eta=1, nthread = 2, silent=1, 
                objective=logregobjattr, eval_metric=evalerror)
  bst <- xgb.train(param, dtrain, num_round, watchlist)
  expect_equal(class(bst), "xgb.Booster")
  expect_equal(length(bst$raw), 1064)
})