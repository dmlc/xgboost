#' Save xgb.DMatrix object to binary file
#' 
#' Save xgb.DMatrix object to binary file
#' 
#' @param DMatrix the DMatrix object
#' @param fname the name of the binary file.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' xgb.DMatrix.save(dtrain, 'xgb.DMatrix.data')
#' dtrain <- xgb.DMatrix('xgb.DMatrix.data')
#' @export
#' 
xgb.DMatrix.save <- function(DMatrix, fname) {
  if (typeof(fname) != "character") {
    stop("xgb.save: fname must be character")
  }
  if (class(DMatrix) == "xgb.DMatrix") {
    .Call("XGDMatrixSaveBinary_R", DMatrix, fname, as.integer(FALSE), 
          PACKAGE = "xgboost")
    return(TRUE)
  }
  stop("xgb.DMatrix.save: the input must be xgb.DMatrix")
  return(FALSE)
} 
