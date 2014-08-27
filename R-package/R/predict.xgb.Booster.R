#' @export
setClass("xgb.Booster")

#' @export
setMethod("predict",
          signature = "xgb.Booster",
          definition = function(object, newdata, outputmargin = FALSE)
          {
              if (class(newdata) != "xgb.DMatrix") {
                  newdata = xgb.DMatrix(newdata)
              }
              ret <- .Call("XGBoosterPredict_R", object, newdata, 
                           as.integer(outputmargin), PACKAGE="xgboost")
              return(ret)
          })

