#' Load xgboost model from binary file
#' 
#' Load xgboost model from the binary model file. 
#' 
#' @param modelfile the name of the binary input file.
#' 
#' @details 
#' The input file is expected to contain a model saved in an xgboost-internal binary format
#' using either \code{\link{xgb.save}} or \code{\link{cb.save.model}} in R, or using some 
#' appropriate methods from other xgboost interfaces. E.g., a model trained in Python and 
#' saved from there in xgboost format, could be loaded from R.
#' 
#' Note: a model saved as an R-object, has to be loaded using corresponding R-methods,
#' not \code{xgb.load}.
#' 
#' @return 
#' An object of \code{xgb.Booster} class.
#' 
#' @seealso 
#' \code{\link{xgb.save}}, \code{\link{xgb.Booster.complete}}. 
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max_depth = 2, 
#'                eta = 1, nthread = 2, nrounds = 2,objective = "binary:logistic")
#' xgb.save(bst, 'xgb.model')
#' bst <- xgb.load('xgb.model')
#' if (file.exists('xgb.model')) file.remove('xgb.model')
#' pred <- predict(bst, test$data)
#' @export
xgb.load <- function(modelfile) {
  if (is.null(modelfile))
    stop("xgb.load: modelfile cannot be NULL")

  handle <- xgb.Booster.handle(modelfile = modelfile)
  # re-use modelfile if it is raw so we do not need to serialize
  if (typeof(modelfile) == "raw") {
    bst <- xgb.handleToBooster(handle, modelfile)
  } else {
    bst <- xgb.handleToBooster(handle, NULL)
  }
  bst <- xgb.Booster.complete(bst, saveraw = TRUE)
  return(bst)
}
