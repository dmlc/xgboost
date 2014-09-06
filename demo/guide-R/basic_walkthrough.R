require(xgboost)

data(iris)
iris[,5] <- as.numeric(iris[,5]=='setosa')
iris <- as.matrix(iris)
set.seed(20)
test_ind <- sample(1:nrow(iris),50)
train_ind <- setdiff(1:nrow(iris),test_ind)
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


cat('start running example of build DMatrix from numpy array\n')
x <- iris[,1:4]
y <- iris[,5]
class(x)
dtrain <- xgb.DMatrix(x, label = y)
bst <- xgb.train(param, dtrain, num_round, watchlist)

cat('start running example of build DMatrix from scipy.sparse CSR Matrix\n')
x <- as(x,'dgCMatrix')
class(x)
dtrain <- xgb.DMatrix(x, label = y)
bst <- xgb.train(param, dtrain, num_round, watchlist)


