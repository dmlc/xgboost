setGeneric("nrow")

#' @param x Object of class \code{xgb.DMatrix}
#' @title \code{nrow} return the number of rows present in x.
#' @rdname nrow
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
