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
#' 
#' @examples
#' data(iris)
#' bst <- xgboost(as.matrix(iris[,1:4]),as.numeric(iris[,5]), nrounds = 2)
#' pred <- predict(bst, as.matrix(iris[,1:4]))
#' @export
#' 
setMethod("predict", signature = "xgb.Booster", 
          definition = function(object, newdata, outputmargin = FALSE) {
  if (class(newdata) != "xgb.DMatrix") {
    newdata <- xgb.DMatrix(newdata)
  }
  ret <- .Call("XGBoosterPredict_R", object, newdata, as.integer(outputmargin), PACKAGE = "xgboost")
  return(ret)
})
 
