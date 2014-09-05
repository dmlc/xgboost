#' Contruct xgb.DMatrix object
#' 
#' Contruct xgb.DMatrix object from dense matrix, sparse matrix or local file.
#' 
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character 
#'   indicating the data file.
#' @param info a list of information of the xgb.DMatrix object
#' @param missing Missing is only used when input is dense matrix, pick a float
#     value that represents missing value. Sometime a data use 0 or other extreme value to represents missing values.
#
#' @param ... other information to pass to \code{info}.
#' 
#' @examples
#' data(iris)
#' iris[,5] <- as.numeric(iris[,5])
#' dtrain <- xgb.DMatrix(as.matrix(iris[,1:4]), label=iris[,5])
#' xgb.DMatrix.save(dtrain, 'iris.xgb.DMatrix')
#' dtrain <- xgb.DMatrix('iris.xgb.DMatrix')
#' @export
#' 
xgb.DMatrix <- function(data, info = list(), missing = 0, ...) {
  if (typeof(data) == "character") {
    handle <- .Call("XGDMatrixCreateFromFile_R", data, as.integer(FALSE), 
                    PACKAGE = "xgboost")
  } else if (is.matrix(data)) {
    handle <- .Call("XGDMatrixCreateFromMat_R", data, missing, 
                    PACKAGE = "xgboost")
  } else if (class(data) == "dgCMatrix") {
    handle <- .Call("XGDMatrixCreateFromCSC_R", data@p, data@i, data@x, 
                    PACKAGE = "xgboost")
  } else {
    stop(paste("xgb.DMatrix: does not support to construct from ", 
               typeof(data)))
  }
  dmat <- structure(handle, class = "xgb.DMatrix")
  
  info <- append(info, list(...))
  if (length(info) == 0) 
    return(dmat)
  for (i in 1:length(info)) {
    p <- info[i]
    xgb.setinfo(dmat, names(p), p[[1]])
  }
  return(dmat)
} 
