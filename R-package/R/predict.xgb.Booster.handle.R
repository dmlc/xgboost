setClass("xgb.Booster.handle")

setMethod("predict", signature = "xgb.Booster.handle", 
          definition = function(object, ...) {
  if (class(object) != "xgb.Booster.handle"){
    stop("predict: model in prediction must be of class xgb.Booster.handle")
  }
  
  bst <- list(handle = object,raw = NULL)
  class(bst) <- 'xgb.Booster'
  bst$raw <- xgb.save.raw(bst$handle)
  
  ret = predict(bst, ...)
  return(ret)
})

