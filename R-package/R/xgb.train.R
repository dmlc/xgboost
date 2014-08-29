#' eXtreme Gradient Boosting Training
#' 
#' The training function of xgboost
#'
#' @param params the list of parameters. Commonly used ones are:
#' \itemize{
#'   \item \code{objective} objective function, common ones are
#'   \itemize{
#'     \item \code{reg:linear} linear regression
#'     \item \code{binary:logistic} logistic regression for classification
#'   }
#'   \item \code{eta} step size of each boosting step
#'   \item \code{max_depth} maximum depth of the tree
#'   \item \code{nthread} number of thread used in training, if not set, all threads are used
#' }
#'
#'   See \url{https://github.com/tqchen/xgboost/wiki/Parameters} for 
#'   further details. See also demo/demo.R for walkthrough example in R.
#' @param dtrain takes an \code{xgb.DMatrix} as the input.
#' @param nrounds the max number of iterations
#' @param watchlist what information should be printed when \code{verbose=1} or
#'   \code{verbose=2}. Watchlist is used to specify validation set monitoring
#'   during training. For example user can specify
#'    watchlist=list(validation1=mat1, validation2=mat2) to watch
#'    the performance of each round's model on mat1 and mat2
#'
#' @param obj customized objective function. Given prediction and dtrain, 
#'   return gradient and second order gradient.
#' @param feval custimized evaluation function. Given prediction and dtrain,
#'   return a \code{list(metric='metric-name', value='metric-value')}.
#' @param ... other parameters to pass to \code{params}.
#' 
#' @details 
#' This is the training function for xgboost.
#'
#' Parallelization is automatically enabled if OpenMP is present.
#' Number of threads can also be manually specified via "nthread" parameter.
#' 
#' This function only accepts an \code{xgb.DMatrix} object as the input.
#' It supports advanced features such as watchlist, customized objective function,
#' therefore it is more flexible than \code{\link{xgboost}}.
#' 
#' 
#' @examples
#' data(iris)
#' iris[,5] <- as.numeric(iris[,5])
#' dtrain <- xgb.DMatrix(as.matrix(iris[,1:4]), label=iris[,5])
#' dtest <- dtrain
#' watchlist <- list(eval = dtest, train = dtrain)
#' param <- list(max_depth = 2, eta = 1, silent = 1)
#' logregobj <- function(preds, dtrain) {
#'    labels <- getinfo(dtrain, "label")
#'    preds <- 1/(1 + exp(-preds))
#'    grad <- preds - labels
#'    hess <- preds * (1 - preds)
#'    return(list(grad = grad, hess = hess))
#' }
#' evalerror <- function(preds, dtrain) {
#'   labels <- getinfo(dtrain, "label")
#'   err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
#'   return(list(metric = "error", value = err))
#' }
#' bst <- xgb.train(param, dtrain, nround = 2, watchlist, logregobj, evalerror)
#' @export
#' 
xgb.train <- function(params=list(), dtrain, nrounds, watchlist = list(), 
                      obj = NULL, feval = NULL, ...) {
  if (typeof(params) != "list") {
    stop("xgb.train: first argument params must be list")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.train: second argument dtrain must be xgb.DMatrix")
  }
  params = append(params, list(...))
  bst <- xgb.Booster(params, append(watchlist, dtrain))
  for (i in 1:nrounds) {
    if (is.null(obj)) {
      succ <- xgb.iter.update(bst, dtrain, i - 1)
    } else {
      pred <- xgb.predict(bst, dtrain)
      gpair <- obj(pred, dtrain)
      succ <- xgb.iter.boost(bst, dtrain, gpair)
    }
    if (length(watchlist) != 0) {
      if (is.null(feval)) {
        msg <- xgb.iter.eval(bst, watchlist, i - 1)
        cat(msg)
        cat("\n")
      } else {
        cat("[")
        cat(i)
        cat("]")
        for (j in 1:length(watchlist)) {
          w <- watchlist[j]
          if (length(names(w)) == 0) {
            stop("xgb.eval: name tag must be presented for every elements in watchlist")
          }
          ret <- feval(xgb.predict(bst, w[[1]]), w[[1]])
          cat("\t")
          cat(names(w))
          cat("-")
          cat(ret$metric)
          cat(":")
          cat(ret$value)
        }
        cat("\n")
      }
    }
  }
  return(bst)
} 
