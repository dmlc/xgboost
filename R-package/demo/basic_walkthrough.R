require(xgboost)
require(methods)
# we load in the agaricus dataset
# In this example, we are aiming to predict whether a mushroom can be eated
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test
# the loaded data is stored in sparseMatrix, and label is a numeric vector in {0,1}
class(train$label)
class(train$data)

#-------------Basic Training using XGBoost-----------------
# this is the basic usage of xgboost you can put matrix in data field
# note: we are puting in sparse matrix here, xgboost naturally handles sparse input
# use sparse matrix when your feature is sparse(e.g. when you using one-hot encoding vector)
print("training xgboost with sparseMatrix")
bst <- xgboost(data = train$data, label = train$label, max.depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic")
# alternatively, you can put in dense matrix, i.e. basic R-matrix
print("training xgboost with Matrix")
bst <- xgboost(data = as.matrix(train$data), label = train$label, max.depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic")

# you can also put in xgb.DMatrix object, stores label, data and other meta datas needed for advanced features
print("training xgboost with xgb.DMatrix")
dtrain <- xgb.DMatrix(data = train$data, label = train$label)
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nround = 2, objective = "binary:logistic")

# Verbose = 0,1,2
print ('train xgboost with verbose 0, no message')
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic", verbose = 0)
print ('train xgboost with verbose 1, print evaluation metric')
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic", verbose = 1)
print ('train xgboost with verbose 2, also print information about tree')
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic", verbose = 2)

# you can also specify data as file path to a LibSVM format input
# since we do not have this file with us, the following line is just for illustration
# bst <- xgboost(data = 'agaricus.train.svm', max.depth = 2, eta = 1, nround = 2,objective = "binary:logistic")

#--------------------basic prediction using xgboost--------------
# you can do prediction using the following line
# you can put in Matrix, sparseMatrix, or xgb.DMatrix 
pred <- predict(bst, test$data)
err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error=", err))

#-------------------save and load models-------------------------
# save model to binary local file
xgb.save(bst, "xgboost.model")
# load binary model to R
bst2 <- xgb.load("xgboost.model")
pred2 <- predict(bst2, test$data)
# pred2 should be identical to pred
print(paste("sum(abs(pred2-pred))=", sum(abs(pred2-pred))))

#----------------Advanced features --------------
# to use advanced features, we need to put data in xgb.DMatrix
dtrain <- xgb.DMatrix(data = train$data, label=train$label)
dtest <- xgb.DMatrix(data = test$data, label=test$label)
#---------------Using watchlist----------------
# watchlist is a list of xgb.DMatrix, each of them tagged with name
watchlist <- list(train=dtrain, test=dtest)
# to train with watchlist, use xgb.train, which contains more advanced features
# watchlist allows us to monitor the evaluation result on all data in the list 
print ('train xgboost using xgb.train with watchlist')
bst <- xgb.train(data=dtrain, max.depth=2, eta=1, nround=2, watchlist=watchlist,
                 objective = "binary:logistic")
# we can change evaluation metrics, or use multiple evaluation metrics
print ('train xgboost using xgb.train with watchlist, watch logloss and error')
bst <- xgb.train(data=dtrain, max.depth=2, eta=1, nround=2, watchlist=watchlist,
                 eval.metric = "error", eval.metric = "logloss",
                 objective = "binary:logistic")

# xgb.DMatrix can also be saved using xgb.DMatrix.save
xgb.DMatrix.save(dtrain, "dtrain.buffer")
# to load it in, simply call xgb.DMatrix
dtrain2 <- xgb.DMatrix("dtrain.buffer")
bst <- xgb.train(data=dtrain2, max.depth=2, eta=1, nround=2, watchlist=watchlist,
                 objective = "binary:logistic")
# information can be extracted from xgb.DMatrix using getinfo
label = getinfo(dtest, "label")
pred <- predict(bst, dtest)
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error=", err))

# Finally, you can dump the tree you learned using xgb.dump into a text file
xgb.dump(bst, "dump.raw.txt")

