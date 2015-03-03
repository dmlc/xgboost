require(xgboost)
# load in the agaricus dataset
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

param <- list(max.depth=2,eta=1,silent=1,objective='binary:logistic')
watchlist <- list(eval = dtest, train = dtrain)
nround = 5

# training the model for two rounds
bst = xgb.train(param, dtrain, nround, nthread = 2, watchlist)
cat('start testing prediction from first n trees\n')

### predict using first 2 tree
pred_with_leaf = predict(bst, dtest, ntreelimit = 2, predleaf = TRUE)
head(pred_with_leaf)
# by default, we predict using all the trees
pred_with_leaf = predict(bst, dtest, predleaf = TRUE)
head(pred_with_leaf)
