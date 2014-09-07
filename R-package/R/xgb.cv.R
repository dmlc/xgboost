#' Cross Validation
#' 
#' The cross valudation function of xgboost
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
#' @param nfold number of folds used
#' @param label option field, when data is Matrix
#' @param showsd boolean, whether show standard deviation of cross validation
#' @param metrics, list of evaluation metrics to be used in corss validation,
#'   when it is not specified, the evaluation metric is chosen according to objective function.
#'   Possible options are:
#' \itemize{
#'   \item \code{error} binary classification error rate
#'   \item \code{rmse} Rooted mean square error
#'   \item \code{logloss} negative log-likelihood function
#'   \item \code{auc} Area under curve
#'   \item \code{merror} Exact matching error, used to evaluate multi-class classification
#' }
#' @param obj customized objective function. Returns gradient and second order 
#'   gradient with given prediction and dtrain, 
#' @param feval custimized evaluation function. Returns 
#'   \code{list(metric='metric-name', value='metric-value')} with given 
#'   prediction and dtrain,
#' @param ... other parameters to pass to \code{params}.
#' 
#' @details 
#' This is the cross validation function for xgboost
#'
#' Parallelization is automatically enabled if OpenMP is present.
#' Number of threads can also be manually specified via "nthread" parameter.
#' 
#' This function only accepts an \code{xgb.DMatrix} object as the input.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
#' history <- xgb.cv(data = dtrain, nround=3, nfold = 5, metrics=list("rmse","auc"),
#'                   "max.depth"=3, "eta"=1, "objective"="binary:logistic")
#' @export
#'
xgb.cv <- function(params=list(), data, nrounds, nfold, label = NULL,
                   showsd = TRUE, metrics=list(), obj = NULL, feval = NULL, ...) {
  if (typeof(params) != "list") {
    stop("xgb.cv: first argument params must be list")
  }
  if (nfold <= 1) {
    stop("nfold must be bigger than 1")
  }
  dtrain <- xgb.get.DMatrix(data, label)
  params <- append(params, list(...))
  params <- append(params, list(silent=1))
  for (mc in metrics) {
    params <- append(params, list("eval_metric"=mc))
  }

  folds <- xgb.cv.mknfold(dtrain, nfold, params)
  history <- list()
  for (i in 1:nrounds) {
    msg <- list()
    for (k in 1:nfold) {
      fd <- folds[[k]]
      succ <- xgb.iter.update(fd$booster, fd$dtrain, i - 1, obj)      
      msg[[k]] <- strsplit(xgb.iter.eval(fd$booster, fd$watchlist, i - 1, feval), 
                           "\t")[[1]]
    }
    ret <- xgb.cv.aggcv(msg, showsd)
    history <- append(history, ret)
    cat(paste(ret, "\n", sep=""))
  }
  return (TRUE)
}
