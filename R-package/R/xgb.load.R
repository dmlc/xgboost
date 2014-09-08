#' Load xgboost model from binary file
#' 
#' Load xgboost model from the binary model file
#' 
#' @param modelfile the name of the binary file.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nround = 2,objective = "binary:logistic")
#' xgb.save(bst, 'xgb.model')
#' bst <- xgb.load('xgb.model')
#' pred <- predict(bst, test$data)
#' @export
#' 
xgb.load <- function(modelfile) {
  if (is.null(modelfile)) 
    stop("xgb.load: modelfile cannot be NULL")
  xgb.Booster(modelfile = modelfile)
} 
