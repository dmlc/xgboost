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
#' bst <- xgboost(data = train$data, label = train$label, max_depth = 2, 
#'                eta = 1, nthread = 2, nrounds = 2,objective = "binary:logistic")
#' raw <- xgb.save.raw(bst)
#' bst <- xgb.load(raw)
#' pred <- predict(bst, test$data)
#'
#' @export
xgb.save.raw <- function(model) {
  model <- xgb.get.handle(model)
  .Call(XGBoosterModelToRaw_R, model)
}
