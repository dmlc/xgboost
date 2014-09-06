require(xgboost)
require(methods)
data(agaricus.train)
data(agaricus.test)

# we use agaricus data as example dataset
# we will show how to use xgboost to do binary classification here

trainX = agaricus.train$data
trainY = agaricus.train$label
testX = agaricus.test$data
testY = agaricus.test$label
#-------------------------------------
# this is the basic usage of xgboost
# you can put sparse matrix in data field. this is helpful when your data is sparse
# for example, when you use one-hot encoding for feature vectors
bst <- xgboost(data = trainX, label = trainY, max_depth = 1, eta = 1, nround = 2,
               objective = "binary:logistic")
# alternatively, you can put dense matrix
denseX <- as(trainX, "matrix")
bst <- xgboost(data = denseX, label = trainY, max_depth = 1, eta = 1, nround = 2,
               objective = "binary:logistic")

# you can also specify data as file path to a LibSVM format input
# since we do not have libsvm format file for iris, next line is only for illustration
# bst <- xgboost(data = 'iris.svm', max_depth = 2, eta = 1, nround = 2, objective = "binary:logistic")

dtrain <- xgb.DMatrix(trainX, label=trainY)
dtest <- xgb.DMatrix(testX, label=testY)


param <- list(max_depth=2,eta=1,silent=1,objective='binary:logistic')
watchlist <- list(eval = dtest, train = dtrain)
num_round <- 2
bst <- xgb.train(param, dtrain, num_round, watchlist)
preds <- predict(bst, dtest)
labels <- getinfo(dtest,'label')
cat('error=', mean(as.numeric(preds>0.5)!=labels),'\n')
xgb.save(bst, 'xgb.model')
xgb.dump(bst, 'dump.raw.txt')
xgb.dump(bst, 'dump.nuce.txt','../data/featmap.txt')

bst2 <- xgb.load('xgb.model')
preds2 <- predict(bst2,dtest)
stopifnot(sum((preds-preds2)^2)==0)

############################ Test xgb.DMatrix with local file, sparse matrix and dense matrix in R.
