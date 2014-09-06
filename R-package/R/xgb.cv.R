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
#'   further details. See also inst/examples/demo.R for walkthrough example in R.
#' @param data takes an \code{xgb.DMatrix} as the input.
#' @param nrounds the max number of iterations
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
#'
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
#' @export
#'
xgb.cv <- function(params=list(), data, nrounds, metrics=list(), label = NULL,
                   obj = NULL, feval = NULL, ...) {
  if (typeof(params) != "list") {
    stop("xgb.cv: first argument params must be list")
  }
  dtrain <- xgb.get.DMatrix(data, label)
  params = append(params, list(...))
  
}
