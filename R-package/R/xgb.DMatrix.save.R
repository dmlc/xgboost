#' Save xgb.DMatrix object to binary file
#' 
#' Save xgb.DMatrix object to binary file
#' 
#' @param DMatrix the model object.
#' @param fname the name of the binary file.
#' 
#' @examples
#' data(iris)
#' iris[,5] <- as.numeric(iris[,5])
#' dtrain <- xgb.DMatrix(as.matrix(iris[,1:4]), label=iris[,5])
#' xgb.DMatrix.save(dtrain, 'iris.xgb.DMatrix')
#' dtrain <- xgb.DMatrix('iris.xgb.DMatrix')
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
  stop("xgb.save: the input must be either xgb.DMatrix or xgb.Booster")
  return(FALSE)
} 
