#' Save xgb.DMatrix object to binary file
#' 
#' Save xgb.DMatrix object to binary file
#' 
#' @param dmatrix the \code{xgb.DMatrix} object
#' @param fname the name of the file to write.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' xgb.DMatrix.save(dtrain, 'xgb.DMatrix.data')
#' dtrain <- xgb.DMatrix('xgb.DMatrix.data')
#' @export
xgb.DMatrix.save <- function(dmatrix, fname) {
  if (typeof(fname) != "character")
    stop("fname must be character")
  if (!inherits(dmatrix, "xgb.DMatrix"))
    stop("dmatrix must be xgb.DMatrix")
  
  .Call(XGDMatrixSaveBinary_R, dmatrix, fname[1], 0L)
  return(TRUE)
}
