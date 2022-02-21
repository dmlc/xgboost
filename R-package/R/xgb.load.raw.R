#' Load serialised xgboost model from R's raw vector
#'
#' User can generate raw memory buffer by calling xgb.save.raw
#'
#' @param buffer the buffer returned by xgb.save.raw
#'
#' @export
xgb.load.raw <- function(buffer, as_booster = TRUE) {
  cachelist <- list()
  handle <- .Call(XGBoosterCreate_R, cachelist)
  .Call(XGBoosterLoadModelFromRaw_R, handle, buffer)
  class(handle) <- "xgb.Booster.handle"

  if (as_booster) {
    booster <- list(handle = handle, raw = NULL)
    class(booster) <- "xgb.Booster"
    booster <- xgb.Booster.complete(booster, saveraw = TRUE)
    return(booster)
  } else {
    return (handle)
  }
}
