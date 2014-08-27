# Main function for xgboost-package

xgboost <- function(data = NULL, label = NULL, params = list(), nrounds = 10, 
                    verbose = 1, ...) {
  inClass <- class(data)
  if (inClass == "dgCMatrix" || inClass == "matrix") {
    if (is.null(label)) 
      stop("xgboost: need label when data is a matrix")
    dtrain <- xgb.DMatrix(data, label = label)
  } else {
    if (!is.null(label)) 
      warning("xgboost: label will be ignored.")
    if (inClass == "character") 
      dtrain <- xgb.DMatrix(data) else if (inClass == "xgb.DMatrix") 
      dtrain <- data else stop("xgboost: Invalid input of data")
  }
  
  if (verbose > 1) 
    silent <- 0 else silent <- 1
  
  params <- append(params, list(silent = silent))
  params <- append(params, list(...))
  
  if (verbose > 0) 
    watchlist <- list(train = dtrain) else watchlist <- list()
  
  bst <- xgb.train(params, dtrain, nrounds, watchlist)
  
  return(bst)
} 
