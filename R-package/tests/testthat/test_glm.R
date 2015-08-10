context('Test generalized linear models')

library(xgboost)

test_that("glm works", {
	data(agaricus.train, package='xgboost')
	data(agaricus.test, package='xgboost')
	dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
	dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)
})
