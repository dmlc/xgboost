setClass('xgb.DMatrix')

#' Get a new DMatrix containing the specified rows of
#' orginal xgb.DMatrix object
#'
#' Get a new DMatrix containing the specified rows of
#' orginal xgb.DMatrix object
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' dsub <- slice(dtrain, 1:3)
#' @rdname slice
#' @export
#' 
slice <- function(object, ...){
    UseMethod("slice")
}

#' @param object Object of class "xgb.DMatrix"
#' @param idxset a integer vector of indices of rows needed
#' @param ... other parameters
#' @rdname slice
#' @method slice xgb.DMatrix
setMethod("slice", signature = "xgb.DMatrix", 
          definition = function(object, idxset, ...) {
              if (class(object) != "xgb.DMatrix") {
                  stop("slice: first argument dtrain must be xgb.DMatrix")
              }
              ret <- .Call("XGDMatrixSliceDMatrix_R", object, idxset, 
                           PACKAGE = "xgboost")
              
              attr_list <- attributes(object)
              nr <- xgb.numrow(object)
              len <- sapply(attr_list,length)
              ind <- which(len==nr)
              if (length(ind)>0) {
                  nms <- names(attr_list)[ind]
                  for (i in 1:length(ind)) {
                    attr(ret,nms[i]) <- attr(object,nms[i])[idxset]
                  }
              }
              return(structure(ret, class = "xgb.DMatrix"))
          })
