require(xgboost)
# load in the agaricus dataset
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

param <- list(max.depth=2,eta=1,silent=1,objective='binary:logistic')
watchlist <- list(eval = dtest, train = dtrain)
nround = 2

# training the model for two rounds
bst = xgb.train(param, dtrain, nround, watchlist)
cat('start testing prediction from first n trees\n')
labels <- getinfo(dtest,'label')

### predict using first 1 tree
ypred1 = predict(bst, dtest, ntreelimit=1)
# by default, we predict using all the trees
ypred2 = predict(bst, dtest)

cat('error of ypred1=', mean(as.numeric(ypred1>0.5)!=labels),'\n')
cat('error of ypred2=', mean(as.numeric(ypred2>0.5)!=labels),'\n')
