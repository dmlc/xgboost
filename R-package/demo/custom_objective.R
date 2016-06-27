require(xgboost)
# load in the agaricus dataset
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

# note: for customized objective function, we leave objective as default
# note: what we are getting is margin value in prediction
# you must know what you are doing
watchlist <- list(eval = dtest, train = dtrain)
num_round <- 2

# user define objective function, given prediction, return gradient and second order gradient
# this is loglikelihood loss
logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

# user defined evaluation function, return a pair metric_name, result
# NOTE: when you do customized loss function, the default prediction value is margin
# this may make buildin evalution metric not function properly
# for example, we are doing logistic loss, the prediction is score before logistic transformation
# the buildin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need write customized evaluation function
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(metric = "error", value = err))
}

param <- list(max_depth=2, eta=1, nthread = 2, silent=1, 
              objective=logregobj, eval_metric=evalerror)
print ('start training with user customized objective')
# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train
bst <- xgb.train(param, dtrain, num_round, watchlist)

#
# there can be cases where you want additional information 
# being considered besides the property of DMatrix you can get by getinfo
# you can set additional information as attributes if DMatrix

# set label attribute of dtrain to be label, we use label as an example, it can be anything 
attr(dtrain, 'label') <- getinfo(dtrain, 'label')
# this is new customized objective, where you can access things you set
# same thing applies to customized evaluation function
logregobjattr <- function(preds, dtrain) {
  # now you can access the attribute in customized function
  labels <- attr(dtrain, 'label')
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}
param <- list(max_depth=2, eta=1, nthread = 2, silent=1, 
              objective=logregobjattr, eval_metric=evalerror)
print ('start training with user customized objective, with additional attributes in DMatrix')
# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train
bst <- xgb.train(param, dtrain, num_round, watchlist)
