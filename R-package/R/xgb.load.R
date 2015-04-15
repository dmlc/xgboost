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
#'                eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")
#' xgb.save(bst, 'xgb.model')
#' bst <- xgb.load('xgb.model')
#' pred <- predict(bst, test$data)
#' @export
#' 
xgb.load <- function(modelfile) {
  if (is.null(modelfile)) 
    stop("xgb.load: modelfile cannot be NULL")
  
  handle <- xgb.Booster(modelfile = modelfile)
  # re-use modelfile if it is raw so we donot need to serialize
  if (typeof(modelfile) == "raw") {
    bst <- xgb.handleToBooster(handle, modelfile)
  } else {
    bst <- xgb.handleToBooster(handle, NULL)
  }
  bst <- xgb.Booster.check(bst)
  return(bst)
} 
