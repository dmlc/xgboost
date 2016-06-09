# Simple interface for training an xgboost model.
# Its documentation is combined with xgb.train.
#
#' @rdname xgb.train
#' @export
xgboost <- function(data = NULL, label = NULL, missing = NA, weight = NULL,
                    params = list(), nrounds,
                    verbose = 1, print.every.n = 1L, 
                    early.stop.round = NULL, maximize = NULL, 
                    save_period = 0, save_name = "xgboost.model",
                    xgb_model = NULL, callbacks = list(), ...) {

  dtrain <- xgb.get.DMatrix(data, label, missing, weight)

  watchlist <- list()
  if (verbose > 0)
    watchlist$train = dtrain

  bst <- xgb.train(params, dtrain, nrounds, watchlist, verbose = verbose, print.every.n=print.every.n,
                   early.stop.round = early.stop.round, maximize = maximize,
                   save_period = save_period, save_name = save_name,
                   xgb_model = xgb_model, callbacks = callbacks, ...)
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

# Various imports
#' @importClassesFrom Matrix dgCMatrix dgeMatrix
#' @importFrom data.table data.table
#' @importFrom data.table as.data.table
#' @importFrom magrittr %>%
#' @importFrom data.table :=
#' @importFrom data.table rbindlist
#' @importFrom stringr str_extract
#' @importFrom stringr str_split
#' @importFrom stringr str_replace
#' @importFrom stringr str_match
#' @import methods
#' @useDynLib xgboost
NULL
