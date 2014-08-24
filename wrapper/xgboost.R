# depends on matrix
succ <- require("Matrix")
if (!succ) {
  stop("xgboost depends on Matrix library")
}
# load in library
dyn.load("./libxgboostR.so")

# constructing DMatrix
xgb.DMatrix <- function(data, info=list(), missing=0.0) {
  if (typeof(data) == "character") {
    handle <- .Call("XGDMatrixCreateFromFile_R", data, as.integer(FALSE))
  } else if(is.matrix(data)) {
    handle <- .Call("XGDMatrixCreateFromMat_R", data, missing)
  } else if(class(data) == "dgCMatrix") {
    handle <- .Call("XGDMatrixCreateFromCSC_R", data@p, data@i, data@x)
  } else {
    stop(paste("xgb.DMatrix: does not support to construct from ", typeof(data)))
  }
  dmat <- structure(handle, class="xgb.DMatrix")
  if (length(info) != 0) {
    for (i in 1:length(info)) {
      p <- info[i]
      xgb.setinfo(dmat, names(p), p[[1]])
    }
  }
  return(dmat)
}
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
  ret <- .Call("XGDMatrixGetInfo_R", dmat, name)
  return(ret)
}
# set information into dmatrix, this mutate dmatrix
xgb.setinfo <- function(dmat, name, info) {
  if (class(dmat) != "xgb.DMatrix") {
    stop("xgb.setinfo: first argument dtrain must be xgb.DMatrix");
  }
  if (name == "label") {
    .Call("XGDMatrixSetInfo_R", dmat, name, as.numeric(info))
    return(TRUE)
  }
  if (name == "weight") {
    .Call("XGDMatrixSetInfo_R", dmat, name, as.numeric(info))
    return(TRUE)
  }
  if (name == "base_margin") {
    .Call("XGDMatrixSetInfo_R", dmat, name, as.numeric(info))
    return(TRUE)
  }
  if (name == "group") {
    .Call("XGDMatrixSetInfo_R", dmat, name, as.integer(info))
    return(TRUE)
  }
  stop(pase("xgb.setinfo: unknown info name", name))
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
  handle <- .Call("XGBoosterCreate_R", cachelist)
  .Call("XGBoosterSetParam_R", handle, "seed", "0")
  if (length(params) != 0) {
    for (i in 1:length(params)) {
      p <- params[i]
      .Call("XGBoosterSetParam_R", handle, names(p), as.character(p))
    }
  }
  if (!is.null(modelfile)) {
    if (typeof(modelfile) != "character"){
      stop("xgb.Booster: modelfile must be character");
    }
    .Call("XGBoosterLoadModel_R", handle, modelfile)
  }
  return(structure(handle, class="xgb.Booster"))
}
# train a model using given parameters
xgb.train <- function(params, dtrain, nrounds=10, watchlist=list(), obj=NULL, feval=NULL) {
  if (typeof(params) != "list") {
    stop("xgb.train: first argument params must be list");
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.train: second argument dtrain must be xgb.DMatrix");
  }
  bst <- xgb.Booster(params, append(watchlist,dtrain))
  for (i in 1:nrounds) {
    if (is.null(obj)) {
      succ <- xgb.iter.update(bst, dtrain, i-1)
    } else {
      pred <- xgb.predict(bst, dtrain)
      gpair <- obj(pred, dtrain)
      succ <- xgb.iter.boost(bst, dtrain, gpair)
    }
    if (length(watchlist) != 0) {
      if (is.null(feval)) {      
        msg <- xgb.iter.eval(bst, watchlist, i-1)
        cat(msg); cat("\n")
      } else {
        cat("["); cat(i); cat("]");
        for (j in 1:length(watchlist)) {
          w <- watchlist[j]
          if (length(names(w)) == 0) {
            stop("xgb.eval: name tag must be presented for every elements in watchlist")
          }
          ret <- feval(xgb.predict(bst, w[[1]]), w[[1]])
          cat("\t"); cat(names(w)); cat("-"); cat(ret$metric); 
          cat(":"); cat(ret$value)
        }
        cat("\n")        
      }
    }
  }
  return(bst)
}
# save model or DMatrix to file 
xgb.save <- function(handle, fname) {
  if (typeof(fname) != "character") {
    stop("xgb.save: fname must be character");
  }
  if (class(handle) == "xgb.Booster") {
    .Call("XGBoosterSaveModel_R", handle, fname);
    return(TRUE)
  }
  if (class(handle) == "xgb.DMatrix") {
    .Call("XGDMatrixSaveBinary_R", handle, fname, as.integer(FALSE))
    return(TRUE)
  }
  stop("xgb.save: the input must be either xgb.DMatrix or xgb.Booster")
  return(FALSE)
}
# predict 
xgb.predict <- function(booster, dmat, outputmargin = FALSE) {
  if (class(booster) != "xgb.Booster") {
    stop("xgb.predict: first argument must be type xgb.Booster")
  }
  if (class(dmat) != "xgb.DMatrix") {
    stop("xgb.predict: second argument must be type xgb.DMatrix")
  }
  ret <- .Call("XGBoosterPredict_R", booster, dmat, as.integer(outputmargin))
  return(ret)
}
# dump model
xgb.dump <- function(booster, fname, fmap = "") {
  if (class(booster) != "xgb.Booster") {
    stop("xgb.dump: first argument must be type xgb.Booster")
  }
  if (typeof(fname) != "character"){
    stop("xgb.dump: second argument must be type character")
  }
  .Call("XGBoosterDumpModel_R", booster, fname, fmap)
  return(TRUE)
}
##--------------------------------------
# the following are low level iteratively function, not needed
# if you do not want to use them
#---------------------------------------
# iteratively update booster with dtrain
xgb.iter.update <- function(booster, dtrain, iter) {
  if (class(booster) != "xgb.Booster") {
    stop("xgb.iter.update: first argument must be type xgb.Booster")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.iter.update: second argument must be type xgb.DMatrix")
  }
  .Call("XGBoosterUpdateOneIter_R", booster, as.integer(iter), dtrain)
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
  .Call("XGBoosterBoostOneIter_R", booster, dtrain, gpair$grad, gpair$hess)
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
  msg <- .Call("XGBoosterEvalOneIter_R", booster, as.integer(iter), watchlist, evnames)
  return(msg)
}
