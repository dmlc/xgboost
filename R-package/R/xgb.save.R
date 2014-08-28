# save model or DMatrix to file
xgb.save <- function(handle, fname) {
  if (typeof(fname) != "character") {
    stop("xgb.save: fname must be character")
  }
  if (class(handle) == "xgb.Booster") {
    .Call("XGBoosterSaveModel_R", handle, fname, PACKAGE = "xgboost")
    return(TRUE)
  }
  stop("xgb.save: the input must be either xgb.DMatrix or xgb.Booster")
  return(FALSE)
} 
