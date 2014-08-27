# dump model
xgb.dump <- function(booster, fname, fmap = "") {
    if (class(booster) != "xgb.Booster") {
        stop("xgb.dump: first argument must be type xgb.Booster")
    }
    if (typeof(fname) != "character"){
        stop("xgb.dump: second argument must be type character")
    }
    .Call("XGBoosterDumpModel_R", booster, fname, fmap, PACKAGE="xgboost")
    return(TRUE)
}
