#' Serialize the booster instance into R's raw vector.  The serialization method differs
#' from \code{\link{xgb.save.raw}} as the latter one saves only the model but not
#' parameters.  This serialization format is not stable across different xgboost versions.
#'
#' @param booster the booster instance
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
#'                eta = 1, nthread = 2, nrounds = 2,objective = "binary:logistic")
#' raw <- xgb.serialize(bst)
#' bst <- xgb.unserialize(raw)
#'
#' @export
xgb.serialize <- function(booster) {
  handle <- xgb.get.handle(booster)
  .Call(XGBoosterSerializeToBuffer_R, handle)
}
