setClass("xgb.Booster.handle")
setClass("xgb.Booster",
         slots = c(handle = "xgb.Booster.handle",
                   raw = "raw"))

#' Predict method for eXtreme Gradient Boosting model
#' 
#' Predicted values based on xgboost model object.
#' 
#' @param object Object of class "xgb.Boost"
#' @param newdata takes \code{matrix}, \code{dgCMatrix}, local data file or 
#'   \code{xgb.DMatrix}. 
#' @param missing Missing is only used when input is dense matrix, pick a float 
#'     value that represents missing value. Sometime a data use 0 or other extreme value to represents missing values.
#' @param outputmargin whether the prediction should be shown in the original
#'   value of sum of functions, when outputmargin=TRUE, the prediction is 
#'   untransformed margin value. In logistic regression, outputmargin=T will
#'   output value before logistic transformation.
#' @param ntreelimit limit number of trees used in prediction, this parameter is
#'  only valid for gbtree, but not for gblinear. set it to be value bigger 
#'  than 0. It will use all trees by default.
#' @param predleaf whether predict leaf index instead. If set to TRUE, the output will be a matrix object.
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")
#' pred <- predict(bst, test$data)
#' @export
#' 
setMethod("predict", signature = "xgb.Booster", 
          definition = function(object, newdata, missing = NULL, 
                                outputmargin = FALSE, ntreelimit = NULL, predleaf = FALSE) {
  if (class(object) != "xgb.Booster"){
    stop("predict: model in prediction must be of class xgb.Booster")
  } else {
    object <- xgb.Booster.check(object, saveraw = FALSE)
  }
  if (class(newdata) != "xgb.DMatrix") {
    if (is.null(missing)) {
      newdata <- xgb.DMatrix(newdata)
    } else {
      newdata <- xgb.DMatrix(newdata, missing = missing)
    }
  }
  if (is.null(ntreelimit)) {
    ntreelimit <- 0
  } else {
    if (ntreelimit < 1){
      stop("predict: ntreelimit must be equal to or greater than 1")
    }
  }
  option = 0
  if (outputmargin) {
    option <- option + 1
  }
  if (predleaf) {
    option <- option + 2
  }
  ret <- .Call("XGBoosterPredict_R", object$handle, newdata, as.integer(option), 
               as.integer(ntreelimit), PACKAGE = "xgboost")
  if (predleaf){
      len <- getinfo(newdata, "nrow")
      if (length(ret) == len){
          ret <- matrix(ret,ncol = 1)
      } else {
          ret <- matrix(ret, ncol = len)
          ret <- t(ret)
      }
  }
  return(ret)
})

