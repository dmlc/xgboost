#' Load serialised XGBoost model from R's raw vector
#'
#' User can generate raw memory buffer by calling [xgb.save.raw()].
#'
#' @param buffer The buffer returned by [xgb.save.raw()].
#' @export
xgb.load.raw <- function(buffer) {
  cachelist <- list()
  bst <- .Call(XGBoosterCreate_R, cachelist)
  .Call(XGBoosterLoadModelFromRaw_R, xgb.get.handle(bst), buffer)
  return(bst)
}
