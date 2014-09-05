#' Load xgboost model from binary file
#' 
#' Load xgboost model from the binary model file
#' 
#' @param modelfile the name of the binary file.
#' 
#' @examples
#' data(iris)
#' bst <- xgboost(as.matrix(iris[,1:4]),as.numeric(iris[,5]), nrounds = 2)
#' xgb.save(bst, 'iris.xgb.model')
#' bst <- xgb.load('iris.xgb.model')
#' pred <- predict(bst, as.matrix(iris[,1:4]))
#' @export
#' 
xgb.load <- function(modelfile) {
  if (is.null(modelfile)) 
    stop("xgb.load: modelfile cannot be NULL")
  xgb.Booster(modelfile = modelfile)
} 
