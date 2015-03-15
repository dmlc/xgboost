#' Set information of an xgb.DMatrix object
#' 
#' Set information of an xgb.DMatrix object
#' 
#' It can be one of the following:
#' 
#' \itemize{
#'     \item \code{label}: label Xgboost learn from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item \code{base_margin}: base margin is the base prediction Xgboost will boost from ;
#'     \item \code{group}.
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
#' @rdname setinfo
#' @export
#' 
setinfo <- function(object, ...){
  UseMethod("setinfo")
}

#' @param object Object of class "xgb.DMatrix"
#' @param name the name of the field to get
#' @param info the specific field of information to set
#' @param ... other parameters
#' @rdname setinfo
#' @method setinfo xgb.DMatrix
setMethod("setinfo", signature = "xgb.DMatrix", 
          definition = function(object, name, info) {
            xgb.setinfo(object, name, info)
          })
