require(xgboost)

context("basic functions")

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test
set.seed(1994)

test_that("train and predict binary classification", {
  nround = 2
  expect_output(
    bst <- xgboost(data = train$data, label = train$label, max.depth = 2,
                  eta = 1, nthread = 2, nround = nround, objective = "binary:logistic")
  , "train-error")
  expect_equal(class(bst), "xgb.Booster")
  
  expect_true(!is.null(bst$evaluation_log))
  expect_equal(nrow(bst$evaluation_log), nround)
  expect_lt(bst$evaluation_log[, min(train_error)], 0.03)
  expect_equal(bst$nboost, bst$ntree)
  
  pred <- predict(bst, test$data)
  expect_equal(length(pred), 1611)
})

test_that("train and predict softprob", {
  expect_output(
    bst <- xgboost(data = as.matrix(iris[, -5]), label = as.numeric(iris$Species) - 1,
                   max.depth = 3, eta = 0.5, nthread = 2, nround = 5,
                   objective = "multi:softprob", num_class=3)
  , "train-merror")
  expect_true(!is.null(bst$evaluation_log))
  expect_lt(bst$evaluation_log[, min(train_merror)], 0.025)
  expect_equal(bst$nboost * 3, bst$ntree)
  
  pred <- predict(bst, as.matrix(iris[, -5]))
  expect_equal(length(pred), nrow(iris) * 3)
})

test_that("train and predict softmax", {
  expect_output(
    bst <- xgboost(data = as.matrix(iris[, -5]), label = as.numeric(iris$Species) - 1,
                   max.depth = 3, eta = 0.15, nthread = 2, nround = 25,
                   objective = "multi:softmax", num_class=3)
  , "train-merror")
  expect_true(!is.null(bst$evaluation_log))
  expect_lt(bst$evaluation_log[, min(train_merror)], 0.025)
  expect_equal(bst$nboost * 3, bst$ntree)
  
  pred <- predict(bst, as.matrix(iris[, -5]))
  expect_equal(length(pred), nrow(iris))
})

test_that("early stopping xgb.train works", {
  expect_output(
    bst <- xgboost(data = train$data, label = train$label, max.depth = 2,
                  eta = 0.3, nthread = 2, nround = 20, objective = "binary:logistic",
                  early.stop.round = 3, maximize = FALSE)
  , "Stopping. Best iteration")
  expect_true(!is.null(bst$best_iteration))
  expect_lt(bst$best_iteration, 19)
  expect_equal(bst$nboost, bst$ntree)
  expect_equal(bst$best_iteration, bst$best_ntreelimit)

  pred <- predict(bst, test$data)
  expect_equal(length(pred), 1611)
})

test_that("training continuation works", {
  dtrain <- xgb.DMatrix(train$data, label = train$label)
  watchlist = list(train=dtrain)
  param <- list(objective = "binary:logistic", max.depth = 2, eta = 1, nthread = 2)

  # for the reference, use 4 iterations at once:
  set.seed(11)
  bst <- xgb.train(param, dtrain, nrounds = 4, watchlist)
  # first two iterations:
  set.seed(11)
  bst1 <- xgb.train(param, dtrain, nrounds = 2, watchlist)
  # continue for two more:
  bst2 <- xgb.train(param, dtrain, nrounds = 2, watchlist, xgb_model = bst1)
  expect_equal(bst$raw, bst2$raw)
  expect_true(!is.null(bst2$evaluation_log))
  expect_equal(dim(bst2$evaluation_log), c(4, 2))
  expect_equal(bst2$evaluation_log, bst$evaluation_log)
  # test continuing from raw model data
  bst2 <- xgb.train(param, dtrain, nrounds = 2, watchlist, xgb_model = bst1$raw)
  expect_equal(bst$raw, bst2$raw)
  expect_equal(dim(bst2$evaluation_log), c(2, 2))
  # test continuing from a model in file
  xgb.save(bst1, "xgboost.model")
  bst2 <- xgb.train(param, dtrain, nrounds = 2, watchlist, xgb_model = "xgboost.model")
  expect_equal(bst$raw, bst2$raw)
  expect_equal(dim(bst2$evaluation_log), c(2, 2))
})


test_that("xgb.cv works", {
  cv <- xgb.cv(data = train$data, label = train$label, max.depth = 2, nfold = 5,
               eta = 1., nthread = 2, nround = 2, objective = "binary:logistic",
               verbose=TRUE)
  expect_is(cv, 'xgb.cv.synchronous')
  expect_true(!is.null(cv$evaluation_log))
  expect_lt(cv$evaluation_log[, min(test_error_mean)], 0.03)
  expect_lt(cv$evaluation_log[, min(test_error_std)], 0.004)
  expect_equal(cv$nboost, cv$ntree)
  expect_true(!is.null(cv$folds) && is.list(cv$folds))
  expect_length(cv$folds, 5)
  expect_true(!is.null(cv$params) && is.list(cv$params))
  expect_true(!is.null(cv$callbacks))
  expect_true(!is.null(cv$call))
})

test_that("early stopping xgb.cv works", {
  expect_output(
    cv <- xgb.cv(data = train$data, label = train$label, max.depth = 2, nfold = 5,
                 eta = 0.5, nthread = 2, nround = 20, objective = "binary:logistic",
                 early.stop.round = 3, maximize = FALSE, verbose=TRUE)
  , "Stopping. Best iteration")
  expect_true(!is.null(cv$best_iteration))
  expect_lt(cv$best_iteration, 19)
  expect_equal(cv$nboost, cv$ntree)
  expect_equal(cv$best_iteration, cv$best_ntreelimit)
})

test_that("prediction in xgb.cv works", {
  nround = 2
  cv <- xgb.cv(data = train$data, label = train$label, max.depth = 2, nfold = 5,
               eta = 0.5, nthread = 2, nround = nround, objective = "binary:logistic",
               verbose=TRUE, prediction = T)
  expect_true(!is.null(cv$evaluation_log))
  expect_true(!is.null(cv$pred))
  expect_length(cv$pred, nrow(train$data))
  err_pred <- sum((cv$pred > 0.5) != train$label)/length(train$label)
  err_log <- cv$evaluation_log[nround, test_error_mean]
  expect_lt(abs(err_pred - err_log), 10e-6)
})

test_that("prediction in early-stopping xgb.cv works", {
  set.seed(123)
  # add some label noise
  lb <- train$label
  lb[sample(length(train$label), 2000)] <- 0
  expect_output(
    cv <- xgb.cv(data = train$data, label = lb, max.depth = 3, nfold = 5,
                 eta = 1., nthread = 2, nround = 20, objective = "binary:logistic",
                 early.stop.round = 3, maximize = FALSE, verbose=TRUE, predict=TRUE)
  , "Stopping. Best iteration")
  expect_true(!is.null(cv$best_iteration))
  expect_lt(cv$best_iteration, 19)
  expect_true(!is.null(cv$evaluation_log))
  expect_true(!is.null(cv$pred))
  expect_length(cv$pred, nrow(train$data))
  err_pred <- sum((cv$pred > 0.5) != lb)/length(lb)
  err_log <- cv$evaluation_log[cv$best_iteration, test_error_mean]
  expect_lt(abs(err_pred - err_log), 10e-6)
})
