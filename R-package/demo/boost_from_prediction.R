require(xgboost)
# load in the agaricus dataset
data(agaricus.train, package = 'xgboost')
data(agaricus.test, package = 'xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

watchlist <- list(eval = dtest, train = dtrain)
###
# advanced: start from a initial base prediction
#
print('start running example to start from a initial prediction')
# train xgboost for 1 round
param <- list(max_depth = 2, eta = 1, nthread = 2, objective = 'binary:logistic')
bst <- xgb.train(param, dtrain, 1, watchlist)
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=TRUE, will always give you margin values before logistic transformation
ptrain <- predict(bst, dtrain, outputmargin = TRUE)
ptest  <- predict(bst, dtest, outputmargin = TRUE)
# set the base_margin property of dtrain and dtest
# base margin is the base prediction we will boost from
setinfo(dtrain, "base_margin", ptrain)
setinfo(dtest, "base_margin", ptest)

print('this is result of boost from initial prediction')
bst <- xgb.train(params = param, data = dtrain, nrounds = 1, watchlist = watchlist)
