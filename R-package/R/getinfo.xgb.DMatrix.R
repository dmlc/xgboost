setClass('xgb.DMatrix')

getinfo <- function(object, ...){
    UseMethod("getinfo")
}

setMethod("getinfo", signature = "xgb.DMatrix", 
          definition = function(object, name) {
              if (typeof(name) != "character") {
                  stop("xgb.getinfo: name must be character")
              }
              if (class(object) != "xgb.DMatrix") {
                  stop("xgb.setinfo: first argument dtrain must be xgb.DMatrix")
              }
              if (name != "label" && name != "weight" && name != "base_margin") {
                  stop(paste("xgb.getinfo: unknown info name", name))
              }
              ret <- .Call("XGDMatrixGetInfo_R", object, name, PACKAGE = "xgboost")
              return(ret)
          })

