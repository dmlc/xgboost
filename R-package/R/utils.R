#' @importClassesFrom Matrix dgCMatrix dgeMatrix
#' @import methods

# depends on matrix
.onLoad <- function(libname, pkgname) {
  library.dynam("xgboost", pkgname, libname)
}
.onUnload <- function(libpath) {
  library.dynam.unload("xgboost", libpath)
}

# set information into dmatrix, this mutate dmatrix
xgb.setinfo <- function(dmat, name, info) {
  if (class(dmat) != "xgb.DMatrix") {
    stop("xgb.setinfo: first argument dtrain must be xgb.DMatrix")
  }
  if (name == "label") {
    .Call("XGDMatrixSetInfo_R", dmat, name, as.numeric(info), 
          PACKAGE = "xgboost")
    return(TRUE)
  }
  if (name == "weight") {
    .Call("XGDMatrixSetInfo_R", dmat, name, as.numeric(info), 
          PACKAGE = "xgboost")
    return(TRUE)
  }
  if (name == "base_margin") {
    .Call("XGDMatrixSetInfo_R", dmat, name, as.numeric(info), 
          PACKAGE = "xgboost")
    return(TRUE)
  }
  if (name == "group") {
    .Call("XGDMatrixSetInfo_R", dmat, name, as.integer(info), 
          PACKAGE = "xgboost")
    return(TRUE)
  }
  stop(paste("xgb.setinfo: unknown info name", name))
  return(FALSE)
}

# construct a Booster from cachelist
xgb.Booster <- function(params = list(), cachelist = list(), modelfile = NULL) {
  if (typeof(cachelist) != "list") {
    stop("xgb.Booster: only accepts list of DMatrix as cachelist")
  }
  for (dm in cachelist) {
    if (class(dm) != "xgb.DMatrix") {
      stop("xgb.Booster: only accepts list of DMatrix as cachelist")
    }
  }
  handle <- .Call("XGBoosterCreate_R", cachelist, PACKAGE = "xgboost")
  if (length(params) != 0) {
    for (i in 1:length(params)) {
      p <- params[i]
      .Call("XGBoosterSetParam_R", handle, names(p), as.character(p), 
            PACKAGE = "xgboost")
    }
  }
  if (!is.null(modelfile)) {
    if (typeof(modelfile) != "character") {
      stop("xgb.Booster: modelfile must be character")
    }
    .Call("XGBoosterLoadModel_R", handle, modelfile, PACKAGE = "xgboost")
  }
  return(structure(handle, class = "xgb.Booster"))
}


# predict, depreciated
xgb.predict <- function(booster, dmat, outputmargin = FALSE) {
  if (class(booster) != "xgb.Booster") {
    stop("xgb.predict: first argument must be type xgb.Booster")
  }
  if (class(dmat) != "xgb.DMatrix") {
    stop("xgb.predict: second argument must be type xgb.DMatrix")
  }
  ret <- .Call("XGBoosterPredict_R", booster, dmat, as.integer(outputmargin), 
               PACKAGE = "xgboost")
  return(ret)
}

## ----the following are low level iteratively function, not needed if
## you do not want to use them ---------------------------------------

# iteratively update booster with dtrain
xgb.iter.update <- function(booster, dtrain, iter) {
  if (class(booster) != "xgb.Booster") {
    stop("xgb.iter.update: first argument must be type xgb.Booster")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.iter.update: second argument must be type xgb.DMatrix")
  }
  .Call("XGBoosterUpdateOneIter_R", booster, as.integer(iter), dtrain, 
        PACKAGE = "xgboost")
  return(TRUE)
}

# iteratively update booster with customized statistics
xgb.iter.boost <- function(booster, dtrain, gpair) {
  if (class(booster) != "xgb.Booster") {
    stop("xgb.iter.update: first argument must be type xgb.Booster")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.iter.update: second argument must be type xgb.DMatrix")
  }
  .Call("XGBoosterBoostOneIter_R", booster, dtrain, gpair$grad, gpair$hess, 
        PACKAGE = "xgboost")
  return(TRUE)
}

# iteratively evaluate one iteration
xgb.iter.eval <- function(booster, watchlist, iter) {
  if (class(booster) != "xgb.Booster") {
    stop("xgb.eval: first argument must be type xgb.Booster")
  }
  if (typeof(watchlist) != "list") {
    stop("xgb.eval: only accepts list of DMatrix as watchlist")
  }
  for (w in watchlist) {
    if (class(w) != "xgb.DMatrix") {
      stop("xgb.eval: watch list can only contain xgb.DMatrix")
    }
  }
  evnames <- list()
  if (length(watchlist) != 0) {
    for (i in 1:length(watchlist)) {
      w <- watchlist[i]
      if (length(names(w)) == 0) {
        stop("xgb.eval: name tag must be presented for every elements in watchlist")
      }
      evnames <- append(evnames, names(w))
    }
  }
  msg <- .Call("XGBoosterEvalOneIter_R", booster, as.integer(iter), watchlist, 
               evnames, PACKAGE = "xgboost")
  return(msg)
} 
