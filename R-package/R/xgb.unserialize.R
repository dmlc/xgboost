#' Load the instance back from \code{\link{xgb.serialize}}
#'
#' @param buffer the buffer containing booster instance saved by \code{\link{xgb.serialize}}
#' @param handle An \code{xgb.Booster.handle} object which will be overwritten with
#' the new deserialized object. Must be a null handle (e.g. when loading the model through
#' `readRDS`). If not provided, a new handle will be created.
#' @return An \code{xgb.Booster.handle} object.
#'
#' @export
xgb.unserialize <- function(buffer, handle = NULL) {
  cachelist <- list()
  if (is.null(handle)) {
    handle <- .Call(XGBoosterCreate_R, cachelist)
  } else {
    if (!is.null.handle(handle))
      stop("'handle' is not null/empty. Cannot overwrite existing handle.")
    .Call(XGBoosterCreateInEmptyObj_R, cachelist, handle)
  }
  tryCatch(
    .Call(XGBoosterUnserializeFromBuffer_R, handle, buffer),
    error = function(e) {
      error_msg <- conditionMessage(e)
      m <- regexec("(src[\\\\/]learner.cc:[0-9]+): Check failed: (header == serialisation_header_)",
                   error_msg, perl = TRUE)
      groups <- regmatches(error_msg, m)[[1]]
      if (length(groups) == 3) {
        stop(paste("It is no longer possible to load models from old versions (1.0.0 or earlier) ",
                   "of XGBoost via readRDS(). To load this model into the latest XGBoost, ",
                   "follow these steps: 1) install the old version of XGBoost that generated ",
                   "this model; 2) load the model using readRDS(); 3) call xgb.save() to export ",
                   "the model; 4) install the latest XGBoost; 5) call xgb.load() to load the ",
                   "model. In general, do NOT use RDS to save XGBoost models. Please use ",
                   "xgb.save() instead to preserve models for the long term. For more ",
                   "information, see ",
                   "https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html",
                   sep = ""))
      } else {
        stop(e)
      }
    })
  class(handle) <- "xgb.Booster.handle"
  return (handle)
}
