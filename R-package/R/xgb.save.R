#' Save xgboost model to binary file
#' 
#' Save xgboost model from xgboost or xgb.train
#' 
#' @param model the model object.
#' @param fname the name of the binary file.
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
xgb.save <- function(model, fname) {
  if (typeof(fname) != "character") {
    stop("xgb.save: fname must be character")
  }
  if (class(model) == "xgb.Booster") {
    .Call("XGBoosterSaveModel_R", model, fname, PACKAGE = "xgboost")
    return(TRUE)
  }
  stop("xgb.save: the input must be xgb.Booster. Use xgb.DMatrix.save to save
       xgb.DMatrix object.")
  return(FALSE)
} 
