require(xgboost)

data(agaricus.train)
data(agaricus.test)

trainX = agaricus.train$data
trainY = agaricus.train$label
testX = agaricus.test$data
testY = agaricus.test$label

dtrain <- xgb.DMatrix(trainX, label=trainY)
dtest <- xgb.DMatrix(testX, label=testY)

num_round <- 2
param <- list(max_depth=2,eta=1,silent=1,objective='binary:logistic')

cat('running cross validation\n')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'error'}, seed = 0)

cat('running cross validation, disable standard deviation display\n')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'error'}, seed = 0, show_stdv = False)

cat('running cross validation, with preprocessing function\n')
# define the preprocessing function
# used to return the preprocessed training, test data, and parameter
# we can use this to do weight rescale, etc.
# as a example, we try to set scale_pos_weight
fpreproc <- function(dtrain, dtest, param){
  label <- getinfo(dtrain, 'label')
  ratio <- mean(label==0)
  param <- append(param, list(scale_pos_weight = ratio))
  return(list(dtrain=dtrain, dtest= dtest, param = param))
}


# do cross validation, for each fold
# the dtrain, dtest, param will be passed into fpreproc
# then the return value of fpreproc will be used to generate
# results of that fold
xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'auc'}, seed = 0, fpreproc = fpreproc)

###
# you can also do cross validation with cutomized loss function
# See custom_objective.py
##
print ('running cross validation, with cutomsized loss function')

logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(metric = "error", value = err))
}

param <- list(max_depth=2,eta=1,silent=1)
# train with customized objective
xgb.cv(param, dtrain, num_round, nfold = 5, seed = 0,
       obj = logregobj, feval=evalerror)

