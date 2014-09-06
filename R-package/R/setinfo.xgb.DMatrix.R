#' Set information of an xgb.DMatrix object
#' 
#' Set information of an xgb.DMatrix object
#' 
#' @examples
#' data(iris)
#' iris[,5] <- as.numeric(iris[,5]=='setosa')
#' dtrain <- xgb.DMatrix(as.matrix(iris[,1:4]), label=iris[,5])
#' labels <- getinfo(dtrain, 'label')
#' setinfo(dtrain, 'label', 1-labels)
#' labels2 <- getinfo(dtrain, 'label')
#' stopifnot(all(labels2 == 1-labels))
#' @rdname setinfo
#' @export
#' 
setinfo <- function(object, ...){
  UseMethod("setinfo")
}

#' @param object Object of class "xgb.DMatrix"
#' @param name the name of the field to get
#' @param info the specific field of information to set
#' @param ... other parameters
#' @rdname setinfo
#' @method setinfo xgb.DMatrix
setMethod("setinfo", signature = "xgb.DMatrix", 
          definition = function(object, name, info) {
            xgb.setinfo(object, name, info)
          })
