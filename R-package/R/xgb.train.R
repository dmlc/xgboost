# train a model using given parameters
xgb.train <- function(params=list(), dtrain, nrounds, watchlist = list(), 
                      obj = NULL, feval = NULL, ...) {
  if (typeof(params) != "list") {
    stop("xgb.train: first argument params must be list")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.train: second argument dtrain must be xgb.DMatrix")
  }
  params = append(params, list(...))
  bst <- xgb.Booster(params, append(watchlist, dtrain))
  for (i in 1:nrounds) {
    if (is.null(obj)) {
      succ <- xgb.iter.update(bst, dtrain, i - 1)
    } else {
      pred <- xgb.predict(bst, dtrain)
      gpair <- obj(pred, dtrain)
      succ <- xgb.iter.boost(bst, dtrain, gpair)
    }
    if (length(watchlist) != 0) {
      if (is.null(feval)) {
        msg <- xgb.iter.eval(bst, watchlist, i - 1)
        cat(msg)
        cat("\n")
      } else {
        cat("[")
        cat(i)
        cat("]")
        for (j in 1:length(watchlist)) {
          w <- watchlist[j]
          if (length(names(w)) == 0) {
            stop("xgb.eval: name tag must be presented for every elements in watchlist")
          }
          ret <- feval(xgb.predict(bst, w[[1]]), w[[1]])
          cat("\t")
          cat(names(w))
          cat("-")
          cat(ret$metric)
          cat(":")
          cat(ret$value)
        }
        cat("\n")
      }
    }
  }
  return(bst)
} 
