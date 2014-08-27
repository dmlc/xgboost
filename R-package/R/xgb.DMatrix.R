# constructing DMatrix
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
