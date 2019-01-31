require(xgboost)
# load in the agaricus dataset
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)
# note: for customized objective function, we leave objective as default
# note: what we are getting is margin value in prediction
# you must know what you are doing
param <- list(max_depth=2, eta=1, nthread=2, verbosity=0)
watchlist <- list(eval = dtest)
num_round <- 20
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
print ('start training with early Stopping setting')

bst <- xgb.train(param, dtrain, num_round, watchlist,
                 objective = logregobj, eval_metric = evalerror, maximize = FALSE,
                 early_stopping_round = 3)
bst <- xgb.cv(param, dtrain, num_round, nfold = 5,
              objective = logregobj, eval_metric = evalerror,
              maximize = FALSE, early_stopping_rounds = 3)
