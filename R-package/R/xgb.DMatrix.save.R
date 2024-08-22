#' Save xgb.DMatrix object to binary file
#'
#' Save xgb.DMatrix object to binary file
#'
#' @param dmatrix the `xgb.DMatrix` object
#' @param fname the name of the file to write.
#'
#' @examples
#' \dontshow{RhpcBLASctl::omp_set_num_threads(1)}
#' data(agaricus.train, package = "xgboost")
#'
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#' fname <- file.path(tempdir(), "xgb.DMatrix.data")
#' xgb.DMatrix.save(dtrain, fname)
#' dtrain <- xgb.DMatrix(fname)
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
