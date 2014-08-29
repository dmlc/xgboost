#' eXtreme Gradient Boosting (Tree) library
#' 
#' A simple interface for xgboost in R
#' 
#' @param data takes \code{matrix}, \code{dgCMatrix}, local data file or 
#'   \code{xgb.DMatrix}. 
#' @param label the response variable. User should not set this field,
#    if data is local data file or  \code{xgb.DMatrix}. 
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
#' @param nrounds the max number of iterations
#' @param verbose If 0, xgboost will stay silent. If 1, xgboost will print 
#'   information of performance. If 2, xgboost will print information of both
#'   performance and construction progress information
#' @param ... other parameters to pass to \code{params}.
#' 
#' @details 
#' This is the modeling function for xgboost.
#' 
#' Parallelization is automatically enabled if OpenMP is present.
#' Number of threads can also be manually specified via "nthread" parameter
#' 
#' @examples
#' data(iris)
#' bst <- xgboost(as.matrix(iris[,1:4]),as.numeric(iris[,5]), nrounds = 2)
#' pred <- predict(bst, as.matrix(iris[,1:4]))
#' @export
#' 
xgboost <- function(data = NULL, label = NULL, params = list(), nrounds, 
                    verbose = 1, ...) {
  inClass <- class(data)
  if (inClass == "dgCMatrix" || inClass == "matrix") {
    if (is.null(label)) 
      stop("xgboost: need label when data is a matrix")
    dtrain <- xgb.DMatrix(data, label = label)
  } else {
    if (!is.null(label)) 
      warning("xgboost: label will be ignored.")
    if (inClass == "character") 
      dtrain <- xgb.DMatrix(data) else if (inClass == "xgb.DMatrix") 
      dtrain <- data else stop("xgboost: Invalid input of data")
  }
  
  if (verbose > 1) {
    silent <- 0 
  } else {
    silent <- 1
  }
  
  params <- append(params, list(silent = silent))
  params <- append(params, list(...))
  
  if (verbose > 0) 
    watchlist <- list(train = dtrain) else watchlist <- list()
  
  bst <- xgb.train(params, dtrain, nrounds, watchlist)
  
  return(bst)
} 
