#' Save xgboost model to R's raw vector,
#' user can call xgb.load to load the model back from raw vector
#' 
#' Save xgboost model from xgboost or xgb.train
#' 
#' @param model the model object.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")
#' raw <- xgb.save.raw(bst)
#' bst <- xgb.load(raw)
#' pred <- predict(bst, test$data)
#' @export
#' 
xgb.save.raw <- function(model) {
  if (class(model) == "xgb.Booster"){
    model <- model$handle
  }
  if (class(model) == "xgb.Booster.handle") {
    raw <- .Call("XGBoosterModelToRaw_R", model, PACKAGE = "xgboost")
    return(raw)
  }
  stop("xgb.raw: the input must be xgb.Booster.handle. Use xgb.DMatrix.save to save
       xgb.DMatrix object.")
}
