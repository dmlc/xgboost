setClass('xgb.DMatrix')

#' Get information of an xgb.DMatrix object
#' 
#' Get information of an xgb.DMatrix object
#' 
#' @examples
#' data(iris)
#' iris[,5] <- as.numeric(iris[,5])
#' dtrain <- xgb.DMatrix(as.matrix(iris[,1:4]), label=iris[,5])
#' labels <- getinfo(dtrain, "label")
#' @rdname getinfo
#' @export
#' 
getinfo <- function(object, ...){
    UseMethod("getinfo")
}

#' @param object Object of class "xgb.DMatrix"
#' @param name the name of the field to get
#' @param ... other parameters
#' @rdname getinfo
#' @method getinfo xgb.DMatrix
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

