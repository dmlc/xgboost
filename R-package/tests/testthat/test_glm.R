context('Test generalized linear models')

library(xgboost)

test_that("glm works", {
	data(agaricus.train, package='xgboost')
	data(agaricus.test, package='xgboost')
	dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
	dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)
	param <- list(objective = "binary:logistic", booster = "gblinear",
              nthread = 2, alpha = 0.0001, lambda = 1)
	watchlist <- list(eval = dtest, train = dtrain)
	num_round <- 2
	bst <- xgb.train(param, dtrain, num_round, watchlist)
	ypred <- predict(bst, dtest)
	labels <- getinfo(dtest, 'label')
})
