#' Save xgboost model to binary file
#' 
#' Save xgboost model from xgboost or xgb.train
#' 
#' @param model the model object.
#' @param fname the name of the file to write.
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
#' pred <- predict(bst, test$data)
#' @export
xgb.save <- function(model, fname) {
  if (typeof(fname) != "character")
    stop("fname must be character")
  if (class(model) != "xgb.Booster")
    stop("the input must be xgb.Booster. Use xgb.DMatrix.save to save xgb.DMatrix object.")
  
  model <- xgb.Booster.check(model, saveraw = FALSE)
  .Call("XGBoosterSaveModel_R", model$handle, fname, PACKAGE = "xgboost")
  return(TRUE)
}
