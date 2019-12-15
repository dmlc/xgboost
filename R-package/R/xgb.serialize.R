#' Serialize the booster instance into R's raw vector.  The serialization method differs
#' from \code{\link{xgb.save.raw}} as the latter one saves only the model but not
#' parameters.  The serialization format is not stable across different xgboost versions.
#'
#' @param booster the booster instance
#'
#' @export
xgb.serialize <- function(booster) {
  handle <- xgb.get.handle(booster)
  .Call(XGBoosterSerializeToBuffer_R, handle)
}
