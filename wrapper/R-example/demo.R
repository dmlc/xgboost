# include xgboost library, must set chdir=TRURE
source('../xgboost.R', chdir=TRUE)

# test code here
dtrain <- xgb.DMatrix("agaricus.txt.train")
dtest <- xgb.DMatrix("agaricus.txt.test")
param = list('bst:max_depth'=2, 'bst:eta'=1, 'silent'=1, 'objective'='binary:logistic')
watchlist <- list('train'=dtrain,'test'=dtest)
# training xgboost model
bst <- xgb.train(param, dtrain, nround=3, watchlist=watchlist)
# make prediction
preds <- xgb.predict(bst, dtest)
labels <- xgb.getinfo(dtest, "label")
err <- as.real(sum(as.integer(preds > 0.5) != labels)) / length(labels)
# print error rate
print(err)

# save dmatrix into binary buffer
succ <- xgb.save(dtest, "dtest.buffer")
# save model into file
succ <- xgb.save(bst, "xgb.model")
# load model in
bst2 <- xgb.Booster(modelfile="xgb.model")
dtest2 <- xgb.DMatrix("dtest.buffer")
preds2 <- xgb.predict(bst2, dtest2)
# print difference
print(sum(abs(preds2-preds)))

###
# advanced: cutomsized loss function
# 
print("start running example to used cutomized objective function")
# note: for customized objective function, we leave objective as default
# note: what we are getting is margin value in prediction
# you must know what you are doing
param <- list('bst:max_depth' = 2, 'bst:eta' = 1, 'silent' =1)
# user define objective function, given prediction, return gradient and second order gradient
# this is loglikelihood loss
logregobj <- function(preds, dtrain) {
  labels <- xgb.getinfo(dtrain, "label")
  preds <- 1.0 / (1.0 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1.0-preds)
  return(list(grad=grad, hess=hess))
}
# user defined evaluation function, return a list(metric="metric-name", value="metric-value")
# NOTE: when you do customized loss function, the default prediction value is margin
# this may make buildin evalution metric not function properly
# for example, we are doing logistic loss, the prediction is score before logistic transformation
# the buildin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need write customized evaluation function
evalerror <- function(preds, dtrain) {
  labels <- xgb.getinfo(dtrain, "label")
  err <- as.real(sum(labels != (preds > 0.0))) / length(labels)
  return(list(metric="error", value=err))
}

# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train
bst <- xgb.train(param, dtrain, nround=2, watchlist, logregobj, evalerror)

###
# advanced: start from a initial base prediction
#
print ('start running example to start from a initial prediction')
# specify parameters via map, definition are same as c++ version
param = list('bst:max_depth'=2, 'bst:eta'=1, 'silent'=1, 'objective'='binary:logistic')
# train xgboost for 1 round
bst <- xgb.train( param, dtrain, 1, watchlist )
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=True, will always give you margin values before logistic transformation
ptrain <- xgb.predict(bst, dtrain, outputmargin=TRUE)
ptest <- xgb.predict(bst, dtest, outputmargin=TRUE)
succ <- xgb.setinfo(dtrain, "base_margin", ptrain)
succ <- xgb.setinfo(dtest, "base_margin", ptest)
print ('this is result of running from initial prediction')
bst <- xgb.train( param, dtrain, 1, watchlist )
