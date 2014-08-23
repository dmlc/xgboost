# load in library
dyn.load("./libxgboostR.so")

# constructing DMatrix
xgb.DMatrix <- function(data) {
  if (typeof(data) == "character") {
    handle <- .Call("XGDMatrixCreateFromFile_R", data, as.integer(FALSE))
  }else {
    stop("xgb.DMatrix cannot recognize data type")
  }
  return(structure(handle, class="xgb.DMatrix"))
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
  for (i in 1:length(params)) {
    p = params[i]    
    .Call("XGBoosterSetParam_R", handle, names(p), as.character(p))
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
xgb.train <- function(params, dtrain, nrounds=10, watchlist=list(), obj=NULL) {
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
      pred = xgb.predict(bst, dtrain)
      gpair = obj(pred, dtrain)
      succ <- xgb.iter.boost(bst, dtrain, gpair)
    }    
    if (length(watchlist) != 0) {
      msg <- xgb.iter.eval(bst, watchlist, i-1)
        cat(msg); cat("\n")
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
    stop("xgb.iter.update: first argument must be type xgb.Booster")
  }
  if (class(dmat) != "xgb.DMatrix") {
    stop("xgb.iter.update: second argument must be type xgb.DMatrix")
  }
  ret = .Call("XGBoosterPredict_R", booster, dmat, as.integer(outputmargin))
  return(ret)
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
  for (i in 1:length(watchlist)) {
    w <- watchlist[i]
    if (length(names(w)) == 0) {
      stop("xgb.eval: name tag must be presented for every elements in watchlist")
    }
    evnames <- append(evnames, names(w))
  }
  msg <- .Call("XGBoosterEvalOneIter_R", booster, as.integer(iter), watchlist, evnames)
  return(msg)
}
