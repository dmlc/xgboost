# get information from dmatrix
xgb.getinfo <- function(dmat, name) {
    if (typeof(name) != "character") {
        stop("xgb.getinfo: name must be character")
    }
    if (class(dmat) != "xgb.DMatrix") {
        stop("xgb.setinfo: first argument dtrain must be xgb.DMatrix");
    }
    if (name != "label" &&
            name != "weight" &&
            name != "base_margin" ) {
        stop(paste("xgb.getinfo: unknown info name", name))
    }
    ret <- .Call("XGDMatrixGetInfo_R", dmat, name, PACKAGE="xgboost")
    return(ret)
}
