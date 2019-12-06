#' Load serialised xgboost model from R's raw vector
#'
#' User can generate raw memory buffer by calling xgb.save.raw
#'
#' @param buffer the buffer returned by xgb.save.raw
#'
#' @export
xgb.load.raw <- function(buffer) {
  cachelist <- list()
  handle <- .Call(XGBoosterCreate_R, cachelist)
  modellst <- unserialize(buffer)
  .Call(XGBoosterLoadModelFromRaw_R, handle, modellst$model)
  .Call(XGBoosterLoadJsonConfig_R, handle, modellst$config)
  class(handle) <- "xgb.Booster.handle"
  return (handle)
}
