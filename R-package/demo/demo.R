require(xgboost)
require(methods)

# helper function to read libsvm format this is very badly written, load in dense, and convert to sparse
# use this only for demo purpose adopted from
# https://github.com/zygmuntz/r-libsvm-format-read-write/blob/master/f_read.libsvm.r
read.libsvm <- function(fname, maxcol) {
  content <- readLines(fname)
  nline <- length(content)
  label <- numeric(nline)
  mat <- matrix(0, nline, maxcol + 1)
  for (i in 1:nline) {
    arr <- as.vector(strsplit(content[i], " ")[[1]])
    label[i] <- as.numeric(arr[[1]])
    for (j in 2:length(arr)) {
      kv <- strsplit(arr[j], ":")[[1]]
      # to avoid 0 index
      findex <- as.integer(kv[1]) + 1
      fvalue <- as.numeric(kv[2])
      mat[i, findex] <- fvalue
    }
  }
  mat <- as(mat, "sparseMatrix")
  return(list(label = label, data = mat))
}

############################ Test xgb.DMatrix with local file, sparse matrix and dense matrix in R.

# Directly read in local file
dtrain <- xgb.DMatrix("agaricus.txt.train")
class(dtrain)

# read file in R
csc <- read.libsvm("agaricus.txt.train", 126)
y <- csc$label
x <- csc$data

# x as Sparse Matrix
class(x)
dtrain <- xgb.DMatrix(x, label = y)

# x as dense matrix
dense.x <- as.matrix(x)
dtrain <- xgb.DMatrix(dense.x, label = y)

############################ Test xgboost with local file, sparse matrix and dense matrix in R.

# Test with DMatrix object
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic")

# Verbose = 0,1,2
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic", verbose = 0)
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic", verbose = 1)
bst <- xgboost(data = dtrain, max_depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic", verbose = 2)

# Test with local file
bst <- xgboost(data = "agaricus.txt.train", max_depth = 2, eta = 1,nround = 2,
               objective = "binary:logistic")

# Test with Sparse Matrix
bst <- xgboost(data = x, label = y, max_depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic")

# Test with dense Matrix
bst <- xgboost(data = dense.x, label = y, max_depth = 2, eta = 1, nround = 2,
               objective = "binary:logistic")


############################ Test predict

# Prediction with DMatrix object
dtest <- xgb.DMatrix("agaricus.txt.test")
pred <- predict(bst, dtest)

# Prediction with local test file
pred <- predict(bst, "agaricus.txt.test")

# Prediction with Sparse Matrix
csc <- read.libsvm("agaricus.txt.test", 126)
test.y <- csc$label
test.x <- csc$data
pred <- predict(bst, test.x)

# Extrac label with getinfo
labels <- getinfo(dtest, "label")
err <- as.numeric(sum(as.integer(pred > 0.5) != labels))/length(labels)
print(paste("error=", err))

############################ Save and load model to hard disk

# save model to binary local file
xgb.save(bst, "xgboost.model")

# load binary model to R
bst <- xgb.load("xgboost.model")
pred <- predict(bst, test.x)

# save model to text file
xgb.dump(bst, "dump.raw.txt")
# save model to text file, with feature map
xgb.dump(bst, "dump.nice.txt", "featmap.txt")

# save a DMatrix object to hard disk
xgb.DMatrix.save(dtrain, "dtrain.buffer")

# load a DMatrix object to R
dtrain <- xgb.DMatrix("dtrain.buffer")

############################ More flexible training function xgb.train

param <- list(max_depth = 2, eta = 1, silent = 1, objective = "binary:logistic")
watchlist <- list(eval = dtest, train = dtrain)

# training xgboost model
bst <- xgb.train(param, dtrain, nround = 2, watchlist = watchlist)

############################ cutomsized loss function

param <- list(max_depth = 2, eta = 1, silent = 1)

# note: for customized objective function, we leave objective as default note: what we are getting is
# margin value in prediction you must know what you are doing

# user define objective function, given prediction, return gradient and second order gradient this is
# loglikelihood loss
logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}
# user defined evaluation function, return a list(metric='metric-name', value='metric-value') NOTE: when
# you do customized loss function, the default prediction value is margin this may make buildin
# evalution metric not function properly for example, we are doing logistic loss, the prediction is
# score before logistic transformation the buildin evaluation error assumes input is after logistic
# transformation Take this in mind when you use the customization, and maybe you need write customized
# evaluation function
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(metric = "error", value = err))
}

# training with customized objective, we can also do step by step training simply look at xgboost.py's
# implementation of train
bst <- xgb.train(param, dtrain, nround = 2, watchlist, logregobj, evalerror)

 
