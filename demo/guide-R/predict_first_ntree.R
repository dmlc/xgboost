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
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)

cat('start testing prediction from first n trees\n')
labels <- getinfo(dtest,'label')
ypred1 = predict(bst, dtest, ntreelimit=1)
ypred2 = predict(bst, dtest)

cat('error of ypred1=', mean(as.numeric(ypred1>0.5)!=labels),'\n')
cat('error of ypred2=', mean(as.numeric(ypred2>0.5)!=labels),'\n')

