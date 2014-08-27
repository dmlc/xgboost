# save model or DMatrix to file 
xgb.DMatrix.save <- function(handle, fname) {
    if (typeof(fname) != "character") {
        stop("xgb.save: fname must be character")
    }
    if (class(handle) == "xgb.DMatrix") {
        .Call("XGDMatrixSaveBinary_R", handle, fname, as.integer(FALSE), PACKAGE="xgboost")
        return(TRUE)
    }
    stop("xgb.save: the input must be either xgb.DMatrix or xgb.Booster")
    return(FALSE)
}
