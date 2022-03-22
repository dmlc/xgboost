# Simple interface for training an xgboost model that wraps \code{xgb.train}.
# Its documentation is combined with xgb.train.
#
#' @rdname xgb.train
#' @export
xgboost <- function(data = NULL, label = NULL, missing = NA, weight = NULL,
                    params = list(), nrounds,
                    verbose = 1, print_every_n = 1L,
                    early_stopping_rounds = NULL, maximize = NULL,
                    save_period = NULL, save_name = "xgboost.model",
                    xgb_model = NULL, callbacks = list(), ...) {
  merged <- check.booster.params(params, ...)
  dtrain <- xgb.get.DMatrix(data, label, missing, weight, nthread = merged$nthread)

  watchlist <- list(train = dtrain)

  bst <- xgb.train(params, dtrain, nrounds, watchlist, verbose = verbose, print_every_n = print_every_n,
                   early_stopping_rounds = early_stopping_rounds, maximize = maximize,
                   save_period = save_period, save_name = save_name,
                   xgb_model = xgb_model, callbacks = callbacks, ...)
  return (bst)
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
#' @importFrom Matrix colSums
#' @importFrom Matrix sparse.model.matrix
#' @importFrom Matrix sparseVector
#' @importFrom Matrix sparseMatrix
#' @importFrom Matrix t
#' @importFrom data.table data.table
#' @importFrom data.table is.data.table
#' @importFrom data.table as.data.table
#' @importFrom data.table :=
#' @importFrom data.table rbindlist
#' @importFrom data.table setkey
#' @importFrom data.table setkeyv
#' @importFrom data.table setnames
#' @importFrom jsonlite fromJSON
#' @importFrom jsonlite toJSON
#' @importFrom utils object.size str tail
#' @importFrom stats predict
#' @importFrom stats median
#' @importFrom utils head
#' @importFrom graphics barplot
#' @importFrom graphics lines
#' @importFrom graphics points
#' @importFrom graphics grid
#' @importFrom graphics par
#' @importFrom graphics title
#' @importFrom grDevices rgb
#'
#' @import methods
#' @useDynLib xgboost, .registration = TRUE
NULL
