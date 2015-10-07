#' eXtreme Gradient Boosting (Tree) library
#' 
#' A simple interface for training xgboost model. Look at \code{\link{xgb.train}} function for a more advanced interface.
#' 
#' @param data takes \code{matrix}, \code{dgCMatrix}, local data file or 
#'   \code{xgb.DMatrix}. 
#' @param label the response variable. User should not set this field,
#'    if data is local data file or  \code{xgb.DMatrix}. 
#' @param params the list of parameters.
#' 
#' Commonly used ones are:
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
#'   Look at \code{\link{xgb.train}} for a more complete list of parameters or \url{https://github.com/dmlc/xgboost/wiki/Parameters} for the full list.
#'   
#'   See also \code{demo/} for walkthrough example in R.
#' 
#' @param nrounds the max number of iterations
#' @param verbose If 0, xgboost will stay silent. If 1, xgboost will print 
#'   information of performance. If 2, xgboost will print information of both
#'   performance and construction progress information
#' @param print.every.n Print every N progress messages when \code{verbose>0}. Default is 1 which means all messages are printed.
#' @param missing Missing is only used when input is dense matrix, pick a float 
#'     value that represents missing value. Sometimes a data use 0 or other extreme value to represents missing values.
#' @param weight a vector indicating the weight for each row of the input.
#' @param early.stop.round If \code{NULL}, the early stopping function is not triggered. 
#'     If set to an integer \code{k}, training with a validation set will stop if the performance 
#'     keeps getting worse consecutively for \code{k} rounds.
#' @param maximize If \code{feval} and \code{early.stop.round} are set, then \code{maximize} must be set as well.
#'     \code{maximize=TRUE} means the larger the evaluation score the better.
#' @param save_period save the model to the disk in every \code{save_period} rounds, 0 means no such action.
#' @param save_name the name or path for periodically saved model file.
#' @param ... other parameters to pass to \code{params}.
#' 
#' @details 
#' This is the modeling function for Xgboost.
#' 
#' Parallelization is automatically enabled if \code{OpenMP} is present.
#' 
#' Number of threads can also be manually specified via \code{nthread} parameter.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
#' pred <- predict(bst, test$data)
#' 
#' @export
#' 
xgboost <- function(data = NULL, label = NULL, missing = NULL, weight = NULL, 
                    params = list(), nrounds, 
                    verbose = 1, print.every.n = 1L, early.stop.round = NULL,
                    maximize = NULL, save_period = 0, save_name = "xgboost.model", ...) {
  dtrain <- xgb.get.DMatrix(data, label, missing, weight)
    
  params <- append(params, list(...))
  
  if (verbose > 0) {
    watchlist <- list(train = dtrain)
  } else {
    watchlist <- list()
  }
  
  bst <- xgb.train(params, dtrain, nrounds, watchlist, verbose = verbose, print.every.n=print.every.n,
                   early.stop.round = early.stop.round, maximize = maximize,
                   save_period = save_period, save_name = save_name)
  
  return(bst)
} 


#' Training part from Mushroom Data Set
#' 
#' This data set is originally from the Mushroom data set,
#' UCI Machine Learning Repository.
#' 
#' This data set includes the following fields:
#' 
#' \itemize{
#'  \item \code{label} the label for each record
#'  \item \code{data} a sparse Matrix of \code{dgCMatrix} class, with 126 columns.
#' }
#'
#' @references
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#' 
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository 
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
#' School of Information and Computer Science.
#' 
#' @docType data
#' @keywords datasets
#' @name agaricus.train
#' @usage data(agaricus.train)
#' @format A list containing a label vector, and a dgCMatrix object with 6513 
#' rows and 127 variables
NULL

#' Test part from Mushroom Data Set
#'
#' This data set is originally from the Mushroom data set,
#' UCI Machine Learning Repository.
#' 
#' This data set includes the following fields:
#' 
#' \itemize{
#'  \item \code{label} the label for each record
#'  \item \code{data} a sparse Matrix of \code{dgCMatrix} class, with 126 columns.
#' }
#'
#' @references
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#' 
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository 
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
#' School of Information and Computer Science.
#' 
#' @docType data
#' @keywords datasets
#' @name agaricus.test
#' @usage data(agaricus.test)
#' @format A list containing a label vector, and a dgCMatrix object with 1611 
#' rows and 126 variables
NULL
