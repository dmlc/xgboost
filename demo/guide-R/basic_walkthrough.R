require(xgboost)
require(methods)
data(iris)
# we use iris data as example dataset
# iris is a dataset with 3 types of iris
# we will show how to use xgboost to do binary classification here
# so the class label will be whether the flower is of type setosa
iris[,5] <- as.numeric(iris[,5]=='setosa')
iris <- as.matrix(iris)
set.seed(20)
# random split train and test set
test_ind <- sample(1:nrow(iris),50)
train_ind <- setdiff(1:nrow(iris),test_ind)
trainX = iris[train_ind,1:4]
trainY = iris[train_ind,5]
testX = iris[train_ind,1:4]
testY = iris[test_ind,5]
#-------------------------------------
# this is the basic usage of xgboost
# you can put matrix in data field
bst <- xgboost(data = trainX, label = trainY, max_depth = 1, eta = 1, nround = 2,
               objective = "binary:logistic")
# alternatively, you can put sparse matrix, this is helpful when your data is sparse
# for example, when you use one-hot encoding for feature vectors
sparseX <- as(trainX, "sparseMatrix")
bst <- xgboost(data = sparseX, label = trainY, max_depth = 1, eta = 1, nround = 2,
               objective = "binary:logistic")

# you can also specify data as file path to a LibSVM format input
# since we do not have libsvm format file for iris, next line is only for illustration
# bst <- xgboost(data = 'iris.svm', max_depth = 2, eta = 1, nround = 2, objective = "binary:logistic")

dtrain <- xgb.DMatrix(iris[train_ind,1:4], label=iris[train_ind,5])
dtest <- xgb.DMatrix(iris[test_ind,1:4], label=iris[test_ind,5])


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
