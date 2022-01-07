#' Save xgboost model to R's raw vector,
#' user can call xgb.load.raw to load the model back from raw vector
#'
#' Save xgboost model from xgboost or xgb.train
#'
#' @param model the model object.
#' @param raw_format The format for encoding the booster.  Available options are
#' \itemize{
#'     \item \code{json}: Encode the booster into JSON text document.
#'     \item \code{ubj}:  Encode the booster into Universal Binary JSON.
#'     \item \code{deprecated}: Encode the booster into old customized binary format.
#' }
#'
#' Right now the default is \code{deprecated} but will be changed to \code{ubj} in upcoming release.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
#'                eta = 1, nthread = 2, nrounds = 2,objective = "binary:logistic")
#' raw <- xgb.save.raw(bst)
#' bst <- xgb.load.raw(raw)
#' pred <- predict(bst, test$data)
#'
#' @export
xgb.save.raw <- function(model, raw_format = "deprecated") {
  handle <- xgb.get.handle(model)
  args <- list(format = raw_format)
  .Call(XGBoosterSaveModelToRaw_R, handle, jsonlite::toJSON(args, auto_unbox = TRUE))
}
