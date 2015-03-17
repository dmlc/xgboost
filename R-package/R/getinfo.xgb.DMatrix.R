setClass('xgb.DMatrix')

#' Get information of an xgb.DMatrix object
#' 
#' Get information of an xgb.DMatrix object
#' 
#' The information can be one of the following:
#' 
#' \itemize{
#'     \item \code{label}: label Xgboost learn from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item \code{base_margin}: base margin is the base prediction Xgboost will boost from ;
#'     \item \code{nrow}: number of rows of the \code{xgb.DMatrix}.
#' }
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' labels <- getinfo(dtrain, 'label')
#' setinfo(dtrain, 'label', 1-labels)
#' labels2 <- getinfo(dtrain, 'label')
#' stopifnot(all(labels2 == 1-labels))
#' @rdname getinfo
#' @export
#' 
getinfo <- function(object, ...){
    UseMethod("getinfo")
}



#' @param object Object of class \code{xgb.DMatrix}
#' @param name the name of the field to get
#' @param ... other parameters
#' @rdname getinfo
#' @method getinfo xgb.DMatrix
setMethod("getinfo", signature = "xgb.DMatrix", 
          definition = function(object, name) {
              if (typeof(name) != "character") {
                  stop("xgb.getinfo: name must be character")
              }
              if (class(object) != "xgb.DMatrix") {
                  stop("xgb.setinfo: first argument dtrain must be xgb.DMatrix")
              }
              if (name != "label" && name != "weight" && 
                      name != "base_margin" && name != "nrow") {
                  stop(paste("xgb.getinfo: unknown info name", name))
              }
              if (name != "nrow"){
                  ret <- .Call("XGDMatrixGetInfo_R", object, name, PACKAGE = "xgboost")
              } else {
                  ret <- xgb.numrow(object)
              }
              return(ret)
          })

