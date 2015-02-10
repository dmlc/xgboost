#' Predict method for eXtreme Gradient Boosting model handle
#' 
#' Predicted values based on xgb.Booster.handle object.
#' 
#' @param object Object of class "xgb.Boost.handle"
#' @param ... Parameters pass to \code{predict.xgb.Booster}
#' 
setMethod("predict", signature = "xgb.Booster.handle", 
          definition = function(object, ...) {
  if (class(object) != "xgb.Booster.handle"){
    stop("predict: model in prediction must be of class xgb.Booster.handle")
  }
  
  bst <- xgb.handleToBooster(object)
  
  ret = predict(bst, ...)
  return(ret)
})

