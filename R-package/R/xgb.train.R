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
#'   \item \code{max.depth} maximum depth of the tree
#'   \item \code{nthread} number of thread used in training, if not set, all threads are used
#' }
#'
#'   See \url{https://github.com/tqchen/xgboost/wiki/Parameters} for 
#'   further details. See also demo/ for walkthrough example in R.
#' @param data takes an \code{xgb.DMatrix} as the input.
#' @param nrounds the max number of iterations
#' @param watchlist what information should be printed when \code{verbose=1} or
#'   \code{verbose=2}. Watchlist is used to specify validation set monitoring
#'   during training. For example user can specify
#'    watchlist=list(validation1=mat1, validation2=mat2) to watch
#'    the performance of each round's model on mat1 and mat2
#'
#' @param obj customized objective function. Returns gradient and second order 
#'   gradient with given prediction and dtrain, 
#' @param feval custimized evaluation function. Returns 
#'   \code{list(metric='metric-name', value='metric-value')} with given 
#'   prediction and dtrain,
#' @param verbose If 0, xgboost will stay silent. If 1, xgboost will print 
#'   information of performance. If 2, xgboost will print information of both
#'
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
#' data(agaricus.train, package='xgboost')
#' dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
#' dtest <- dtrain
#' watchlist <- list(eval = dtest, train = dtrain)
#' param <- list(max.depth = 2, eta = 1, silent = 1)
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
xgb.train <- function(params=list(), data, nrounds, watchlist = list(), 
                      obj = NULL, feval = NULL, verbose = 1, ...) {
  dtrain <- data
  if (typeof(params) != "list") {
    stop("xgb.train: first argument params must be list")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.train: second argument dtrain must be xgb.DMatrix")
  }
  if (verbose > 1) {
    params <- append(params, list(silent = 0))
  } else {
    params <- append(params, list(silent = 1))
  }
  if (length(watchlist) != 0 && verbose == 0) {
    warning('watchlist is provided but verbose=0, no evaluation information will be printed')
    watchlist <- list()
  }
  params = append(params, list(...))
  
  bst <- xgb.Booster(params, append(watchlist, dtrain))
  for (i in 1:nrounds) {
    succ <- xgb.iter.update(bst, dtrain, i - 1, obj)
    if (length(watchlist) != 0) {
      msg <- xgb.iter.eval(bst, watchlist, i - 1, feval)
      cat(paste(msg, "\n", sep=""))
    }
  }
  return(bst)
} 
