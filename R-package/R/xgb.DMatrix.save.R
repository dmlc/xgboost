#' Save xgb.DMatrix object to binary file
#'
#' Save xgb.DMatrix object to binary file
#'
#' @param dmatrix the \code{xgb.DMatrix} object
#' @param fname the name of the file to write.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#' xgb.DMatrix.save(dtrain, 'xgb.DMatrix.data')
#' dtrain <- xgb.DMatrix('xgb.DMatrix.data')
#' if (file.exists('xgb.DMatrix.data')) file.remove('xgb.DMatrix.data')
#' @export
xgb.DMatrix.save <- function(dmatrix, fname) {
  if (typeof(fname) != "character")
    stop("fname must be character")
  if (!inherits(dmatrix, "xgb.DMatrix"))
    stop("dmatrix must be xgb.DMatrix")

  fname <- path.expand(fname)
  .Call(XGDMatrixSaveBinary_R, dmatrix, fname[1], 0L)
  return(TRUE)
}
