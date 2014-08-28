xgb.load <- function(modelfile) {
  if (is.null(modelfile)) 
    stop("xgb.load: modelfile cannot be NULL")
  xgb.Booster(modelfile = modelfile)
} 
