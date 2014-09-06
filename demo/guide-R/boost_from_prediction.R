require(xgboost)

data(agaricus.train)
data(agaricus.test)

trainX = agaricus.train$data
trainY = agaricus.train$label
testX = agaricus.test$data
testY = agaricus.test$label

dtrain <- xgb.DMatrix(trainX, label=trainY)
dtest <- xgb.DMatrix(testX, label=testY)


watchlist <- list(eval = dtest, train = dtrain)
print('start running example to start from a initial prediction\n')
param <- list(max_depth=2,eta=1,silent=1,objective='binary:logistic')
bst <- xgb.train( param, dtrain, 1, watchlist )

ptrain <- predict(bst, dtrain, outputmargin=TRUE)
ptest  <- predict(bst, dtest, outputmargin=TRUE)
# dtrain.set_base_margin(ptrain)
# dtest.set_base_margin(ptest)


cat('this is result of running from initial prediction\n')
bst <- xgb.train( param, dtrain, 1, watchlist )




