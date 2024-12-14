#' Save XGBoost model to R's raw vector
#'
#' Save XGBoost model from [xgboost()] or [xgb.train()].
#' Call [xgb.load.raw()] to load the model back from raw vector.
#'
#' @param model The model object.
#' @param raw_format The format for encoding the booster:
#'   - "json": Encode the booster into JSON text document.
#'   - "ubj":  Encode the booster into Universal Binary JSON.
#'   - "deprecated": Encode the booster into old customized binary format.
#'
#' @examples
#' \dontshow{RhpcBLASctl::omp_set_num_threads(1)}
#' data(agaricus.train, package = "xgboost")
#' data(agaricus.test, package = "xgboost")
#'
#' ## Keep the number of threads to 1 for examples
#' nthread <- 1
#' data.table::setDTthreads(nthread)
#'
#' train <- agaricus.train
#' test <- agaricus.test
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(train$data, label = train$label),
#'   nrounds = 2,
#'   params = xgb.params(
#'     max_depth = 2,
#'     nthread = nthread,
#'     objective = "binary:logistic"
#'   )
#' )
#'
#' raw <- xgb.save.raw(bst)
#' bst <- xgb.load.raw(raw)
#'
#' @export
xgb.save.raw <- function(model, raw_format = "ubj") {
  handle <- xgb.get.handle(model)
  args <- list(format = raw_format)
  .Call(XGBoosterSaveModelToRaw_R, handle, jsonlite::toJSON(args, auto_unbox = TRUE))
}
