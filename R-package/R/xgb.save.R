#' Save xgboost model to binary file
#' 
#' Save xgboost model to a file in binary format.
#' 
#' @param model model object of \code{xgb.Booster} class.
#' @param fname name of the file to write.
#' 
#' @details 
#' This methods allows to save a model in an xgboost-internal binary format which is universal 
#' among the various xgboost interfaces. In R, the saved model file could be read-in later
#' using either the \code{\link{xgb.load}} function or the \code{xgb_model} parameter 
#' of \code{\link{xgb.train}}.
#' 
#' Note: a model can also be saved as an R-object (e.g., by using \code{\link[base]{readRDS}} 
#' or \code{\link[base]{save}}). However, it would then only be compatible with R, and 
#' corresponding R-methods would need to be used to load it.
#' 
#' @seealso 
#' \code{\link{xgb.load}}, \code{\link{xgb.Booster.complete}}. 
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
xgb.save <- function(model, fname) {
  if (typeof(fname) != "character")
    stop("fname must be character")
  if (!inherits(model, "xgb.Booster")) {
    stop("model must be xgb.Booster.",
         if (inherits(model, "xgb.DMatrix")) " Use xgb.DMatrix.save to save an xgb.DMatrix object." else "")
  }
  model <- xgb.Booster.complete(model, saveraw = FALSE)
  .Call(XGBoosterSaveModel_R, model$handle, fname[1])
  return(TRUE)
}
