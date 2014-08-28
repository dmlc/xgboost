setClass('xgb.DMatrix')

#' Get a new DMatrix containing the specified rows of
#' orginal xgb.DMatrix object
#'
#' Get a new DMatrix containing the specified rows of
#' orginal xgb.DMatrix object
#' 
#' @param object Object of class "xgb.DMatrix"
#' @param idxset a integer vector of indices of rows needed
#' 
#' @examples
#' data(iris)
#' iris[,5] <- as.numeric(iris[,5])
#' dtrain <- xgb.DMatrix(as.matrix(iris[,1:4]), label=iris[,5])
#' dsub <- slice(dtrain, c(1,2,3))
#' @export
#' 
slice <- function(object, ...){
    UseMethod("slice")
}

setMethod("slice", signature = "xgb.DMatrix", 
          definition = function(object, idxset) {
              if (class(object) != "xgb.DMatrix") {
                  stop("slice: first argument dtrain must be xgb.DMatrix")
              }
              ret <- .Call("XGDMatrixSliceDMatrix_R", object, idxset, PACKAGE = "xgboost")
              return(structure(ret, class = "xgb.DMatrix"))
          })
