#' Save xgboost model to binary file
#' 
#' Save xgboost model from xgboost or xgb.train
#' 
#' @param model the model object.
#' @param fname the name of the binary file.
#' 
#' @examples
#' data(iris)
#' bst <- xgboost(as.matrix(iris[,1:4]),as.numeric(iris[,5]), nrounds = 2)
#' xgb.save(bst, 'iris.xgb.model')
#' bst <- xgb.load('iris.xgb.model')
#' pred <- predict(bst, as.matrix(iris[,1:4]))
#' @export
#' 
xgb.save <- function(model, fname) {
  if (typeof(fname) != "character") {
    stop("xgb.save: fname must be character")
  }
  if (class(model) == "xgb.Booster") {
    .Call("XGBoosterSaveModel_R", model, fname, PACKAGE = "xgboost")
    return(TRUE)
  }
  stop("xgb.save: the input must be xgb.Booster. Use xgb.DMatrix.save to save
       xgb.DMatrix object.")
  return(FALSE)
} 
