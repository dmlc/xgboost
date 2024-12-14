#' Save XGBoost model to binary file
#'
#' Save XGBoost model to a file in binary or JSON format.
#'
#' @param model Model object of `xgb.Booster` class.
#' @param fname Name of the file to write. Its extension determines the serialization format:
#'   - ".ubj": Use the universal binary JSON format (recommended).
#'     This format uses binary types for e.g. floating point numbers, thereby preventing any loss
#'     of precision when converting to a human-readable JSON text or similar.
#'   - ".json": Use plain JSON, which is a human-readable format.
#'   - ".deprecated": Use **deprecated** binary format. This format will
#'     not be able to save attributes introduced after v1 of XGBoost, such as the "best_iteration"
#'     attribute that boosters might keep, nor feature names or user-specifiec attributes.
#'   - If the format is not specified by passing one of the file extensions above, will
#'     default to UBJ.
#'
#' @details
#'
#' This methods allows to save a model in an XGBoost-internal binary or text format which is universal
#' among the various xgboost interfaces. In R, the saved model file could be read later
#' using either the [xgb.load()] function or the `xgb_model` parameter of [xgb.train()].
#'
#' Note: a model can also be saved as an R object (e.g., by using [readRDS()]
#' or [save()]). However, it would then only be compatible with R, and
#' corresponding R methods would need to be used to load it. Moreover, persisting the model with
#' [readRDS()] or [save()] might cause compatibility problems in
#' future versions of XGBoost. Consult [a-compatibility-note-for-saveRDS-save] to learn
#' how to persist models in a future-proof way, i.e., to make the model accessible in future
#' releases of XGBoost.
#'
#' @seealso [xgb.load()]
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
#' fname <- file.path(tempdir(), "xgb.ubj")
#' xgb.save(bst, fname)
#' bst <- xgb.load(fname)
#' @export
xgb.save <- function(model, fname) {
  if (typeof(fname) != "character")
    stop("fname must be character")
  if (!inherits(model, "xgb.Booster")) {
    stop("model must be xgb.Booster.",
         if (inherits(model, "xgb.DMatrix")) " Use xgb.DMatrix.save to save an xgb.DMatrix object." else "")
  }
  fname <- path.expand(fname)
  .Call(XGBoosterSaveModel_R, xgb.get.handle(model), enc2utf8(fname[1]))
  return(TRUE)
}
