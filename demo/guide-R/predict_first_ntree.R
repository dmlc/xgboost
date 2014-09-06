require(xgboost)

data(agaricus.train)
data(agaricus.test)

trainX = agaricus.train$data
trainY = agaricus.train$label
testX = agaricus.test$data
testY = agaricus.test$label

dtrain <- xgb.DMatrix(trainX, label=trainY)
dtest <- xgb.DMatrix(testX, label=testY)


param <- list(max_depth=2,eta=1,silent=1,objective='binary:logistic')
watchlist <- list(eval = dtest, train = dtrain)
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)

cat('start testing prediction from first n trees\n')
labels <- getinfo(dtest,'label')
ypred1 = predict(bst, dtest, ntreelimit=1)
ypred2 = predict(bst, dtest)

cat('error of ypred1=', mean(as.numeric(ypred1>0.5)!=labels),'\n')
cat('error of ypred2=', mean(as.numeric(ypred2>0.5)!=labels),'\n')

