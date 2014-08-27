# include xgboost library, must set chdir=TRURE
source("../xgboost.R", chdir=TRUE)

# helper function to read libsvm format
# this is very badly written, load in dense, and convert to sparse
# use this only for demo purpose
# adopted from https://github.com/zygmuntz/r-libsvm-format-read-write/blob/master/f_read.libsvm.r
read.libsvm <- function(fname, maxcol) {
  content <- readLines(fname)
  nline <- length(content)
  label <- numeric(nline)
  mat <- matrix(0, nline, maxcol+1)
  for (i in 1:nline) {
    arr <- as.vector(strsplit(content[i], " ")[[1]])
    label[i] <- as.numeric(arr[[1]])
    for (j in 2:length(arr)) {
      kv <- strsplit(arr[j], ":")[[1]]
      # to avoid 0 index
      findex <- as.integer(kv[1]) + 1
      fvalue <- as.numeric(kv[2])
      mat[i,findex] <- fvalue
    }
  }
  mat <- as(mat, "sparseMatrix")
  return(list(label=label, data=mat))
}

# test code here
dtrain <- xgb.DMatrix("agaricus.txt.train")
dtest <- xgb.DMatrix("agaricus.txt.test")
param = list("bst:max_depth"=2, "bst:eta"=1, "silent"=1, "objective"="binary:logistic")
watchlist <- list("eval"=dtest,"train"=dtrain)
# training xgboost model
bst <- xgb.train(param, dtrain, nround=2, watchlist=watchlist)
# make prediction
preds <- xgb.predict(bst, dtest)
labels <- xgb.getinfo(dtest, "label")
err <- as.numeric(sum(as.integer(preds > 0.5) != labels)) / length(labels)
# print error rate
print(paste("error=",err))

# dump model
xgb.dump(bst, "dump.raw.txt")
# dump model with feature map
xgb.dump(bst, "dump.nice.txt", "featmap.txt")

# save dmatrix into binary buffer
succ <- xgb.save(dtest, "dtest.buffer")
# save model into file
succ <- xgb.save(bst, "xgb.model")
# load model and data in 
bst2 <- xgb.Booster(modelfile="xgb.model")
dtest2 <- xgb.DMatrix("dtest.buffer")
preds2 <- xgb.predict(bst2, dtest2)
# assert they are the same
stopifnot(sum(abs(preds2-preds)) == 0)

###
# build dmatrix from sparseMatrix
###
print ('start running example of build DMatrix from R.sparseMatrix')
csc <- read.libsvm("agaricus.txt.train", 126)
label <- csc$label
data <- csc$data
dtrain <- xgb.DMatrix(data, info=list(label=label) )
watchlist <- list("eval"=dtest,"train"=dtrain)
bst <- xgb.train(param, dtrain, nround=2, watchlist=watchlist)

###
# build dmatrix from dense matrix
###
print ('start running example of build DMatrix from R.Matrix')
mat = as.matrix(data)
dtrain <- xgb.DMatrix(mat, info=list(label=label) )
watchlist <- list("eval"=dtest,"train"=dtrain)
bst <- xgb.train(param, dtrain, nround=2, watchlist=watchlist)

###
# advanced: cutomsized loss function
# 
print("start running example to used cutomized objective function")
# note: for customized objective function, we leave objective as default
# note: what we are getting is margin value in prediction
# you must know what you are doing
param <- list("bst:max_depth" = 2, "bst:eta" = 1, "silent" =1)
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
  err <- as.numeric(sum(labels != (preds > 0.0))) / length(labels)
  return(list(metric="error", value=err))
}

# training with customized objective, we can also do step by step training
# simply look at xgboost.py"s implementation of train
bst <- xgb.train(param, dtrain, nround=2, watchlist, logregobj, evalerror)

###
# advanced: start from a initial base prediction
#
print ("start running example to start from a initial prediction")
# specify parameters via map, definition are same as c++ version
param = list("bst:max_depth"=2, "bst:eta"=1, "silent"=1, "objective"="binary:logistic")
# train xgboost for 1 round
bst <- xgb.train( param, dtrain, 1, watchlist )
# Note: we need the margin value instead of transformed prediction in set_base_margin
# do predict with output_margin=True, will always give you margin values before logistic transformation
ptrain <- xgb.predict(bst, dtrain, outputmargin=TRUE)
ptest <- xgb.predict(bst, dtest, outputmargin=TRUE)
succ <- xgb.setinfo(dtrain, "base_margin", ptrain)
succ <- xgb.setinfo(dtest, "base_margin", ptest)
print ("this is result of running from initial prediction")
bst <- xgb.train( param, dtrain, 1, watchlist )
