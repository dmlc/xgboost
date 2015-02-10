setClass("xgb.Booster.handle")

setMethod("predict", signature = "xgb.Booster.handle", 
          definition = function(object, ...) {
  if (class(object) != "xgb.Booster.handle"){
    stop("predict: model in prediction must be of class xgb.Booster.handle")
  }
  
  bst <- xgb.handleToBooster(object)
  # Avoid save a handle without update
  # bst$raw <- xgb.save.raw(object)
  
  ret = predict(bst, ...)
  return(ret)
})

