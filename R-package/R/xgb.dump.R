#' Save xgboost model to text file
#' 
#' Save a xgboost model to text file. Could be parsed later.
#' 
#' @param model the model object.
#' @param fname the name of the binary file.
#' @param fmap feature map file representing the type of feature, to make it
#'        look nice, run demo/demo.R for result and demo/featmap.txt for example
#'        Format: https://github.com/tqchen/xgboost/wiki/Binary-Classification#dump-model
#'
#' @examples
#' data(iris)
#' bst <- xgboost(as.matrix(iris[,1:4]),as.numeric(iris[,5]), nrounds = 2)
#' xgb.dump(bst, 'iris.xgb.model.dump')
#' @export
#' 
xgb.dump <- function(model, fname, fmap = "") {
  if (class(model) != "xgb.Booster") {
    stop("xgb.dump: first argument must be type xgb.Booster")
  }
  if (typeof(fname) != "character") {
    stop("xgb.dump: second argument must be type character")
  }
  .Call("XGBoosterDumpModel_R", model, fname, fmap, PACKAGE = "xgboost")
  return(TRUE)
} 
