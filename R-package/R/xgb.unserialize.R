#' Load the instance back from \code{\link{xgb.serialize}}
#'
#' @param buffer the buffer containing booster instance saved by \code{\link{xgb.serialize}}
#'
#' @export
xgb.unserialize <- function(buffer) {
  cachelist <- list()
  handle <- .Call(XGBoosterCreate_R, cachelist)
  .Call(XGBoosterUnserializeFromBuffer_R, handle, buffer)
  class(handle) <- "xgb.Booster.handle"
  return (handle)
}
