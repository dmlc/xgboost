require(xgboost)

data(iris)
iris[,5] <- as.numeric(iris[,5]=='setosa')
iris <- as.matrix(iris)
set.seed(20)
test_ind <- sample(1:nrow(iris),50)
train_ind <- setdiff(1:nrow(iris),test_ind)
dtrain <- xgb.DMatrix(iris[train_ind,1:4], label=iris[train_ind,5])
dtest <- xgb.DMatrix(iris[test_ind,1:4], label=iris[test_ind,5])


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




