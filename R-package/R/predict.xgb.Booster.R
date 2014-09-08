setClass("xgb.Booster")

#' Predict method for eXtreme Gradient Boosting model
#' 
#' Predicted values based on xgboost model object.
#' 
#' @param object Object of class "xgb.Boost"
#' @param newdata takes \code{matrix}, \code{dgCMatrix}, local data file or 
#'   \code{xgb.DMatrix}. 
#' @param outputmargin whether the prediction should be shown in the original
#'   value of sum of functions, when outputmargin=TRUE, the prediction is 
#'   untransformed margin value. In logistic regression, outputmargin=T will
#'   output value before logistic transformation.
#' @param ntreelimit limit number of trees used in prediction, this parameter is
#'  only valid for gbtree, but not for gblinear. set it to be value bigger 
#'  than 0. It will use all trees by default.
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nround = 2,objective = "binary:logistic")
#' pred <- predict(bst, test$data)
#' @export
#' 
setMethod("predict", signature = "xgb.Booster", 
          definition = function(object, newdata, outputmargin = FALSE, ntreelimit = NULL) {
  if (class(newdata) != "xgb.DMatrix") {
    newdata <- xgb.DMatrix(newdata)
  }
  if (is.null(ntreelimit)) {
    ntreelimit <- 0
  } else {
    if (ntreelimit < 1){
      stop("predict: ntreelimit must be equal to or greater than 1")
    }
  }
  ret <- .Call("XGBoosterPredict_R", object, newdata, as.integer(outputmargin), as.integer(ntreelimit), PACKAGE = "xgboost")
  return(ret)
})
 
