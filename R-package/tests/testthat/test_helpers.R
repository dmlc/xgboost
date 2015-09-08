context('Test helper functions')

require(xgboost)

data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test

bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
               eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")

test_that("dump works", {
  capture.output(print(xgb.dump(bst)))
})


