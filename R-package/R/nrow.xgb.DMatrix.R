setGeneric("nrow")

#' @title Number of xgb.DMatrix rows
#' @description \code{nrow} return the number of rows present in the \code{xgb.DMatrix}.
#' @param x Object of class \code{xgb.DMatrix}
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' stopifnot(nrow(dtrain) == nrow(train$data))
#' 
#' @export
setMethod("nrow",
          signature = "xgb.DMatrix",
          definition = function(x) {
            xgb.numrow(x)
          }
)
