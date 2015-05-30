require(xgboost)
# load in the agaricus dataset
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)

nround <- 2
param <- list(max.depth=2,eta=1,silent=1,nthread = 2, objective='binary:logistic')

cat('running cross validation\n')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
xgb.cv(param, dtrain, nround, nfold=5, metrics={'error'})

cat('running cross validation, disable standard deviation display\n')
# do cross validation, this will print result out as
# [iteration]  metric_name:mean_value+std_value
# std_value is standard deviation of the metric
xgb.cv(param, dtrain, nround, nfold=5,
       metrics={'error'}, showsd = FALSE)

###
# you can also do cross validation with cutomized loss function
# See custom_objective.R
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

param <- list(max.depth=2,eta=1,silent=1,
              objective = logregobj, eval_metric = evalerror)
# train with customized objective
xgb.cv(param, dtrain, nround, nfold = 5)

# do cross validation with prediction values for each fold
res <- xgb.cv(param, dtrain, nround, nfold=5, prediction = TRUE)
res$dt
length(res$pred)
