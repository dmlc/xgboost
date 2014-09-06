#' Save xgboost model to text file
#' 
#' Save a xgboost model to text file. Could be parsed later.
#' 
#' @param model the model object.
#' @param fname the name of the binary file.
#' @param fmap feature map file representing the type of feature. 
#'        Detailed description could be found at 
#'        \url{https://github.com/tqchen/xgboost/wiki/Binary-Classification#dump-model}.
#'        Run inst/examples/demo.R for the result and inst/examples/featmap.txt 
#'        for example Format.
#'        
#'
#' @examples
#' data(iris)
#' bst <- xgboost(as.matrix(iris[,1:4]),as.numeric(iris[,5]=='setosa'), nrounds = 2)
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
