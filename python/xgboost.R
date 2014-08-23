# load in library
dyn.load("libxgboostR.so")

# constructing DMatrix
xgb.DMatrix <- function(data) {
  if (typeof(data) == "character") {
    handle <- .Call("XGDMatrixCreateFromFile_R", data)
  } else {
    stop("xgb.DMatrix cannot recognize data type")
  }
  return(structure(handle, class="xgb.DMatrix"))
}

# construct a Booster from cachelist
xgb.Booster <- function(cachelist, params) {
  if (typeof(cachelist) != "list") {
    stop("xgb.Booster: only accepts list of DMatrix as cachelist")
  }
  for (dm in cachelist) {
    if (class(dm) != "xgb.DMatrix") {
      stop("xgb.Booster: only accepts list of DMatrix as cachelist")
    }
  }
  handle <- .Call("XGBoosterCreate_R", cachelist)
  .Call("XGBoosterSetParam_R", handle, "silent", "1")
  for (i in 1:length(params)) {
    p = params[i]    
    .Call("XGBoosterSetParam_R", handle, names(p), as.character(p))
  }
  return(structure(handle, class="xgb.Booster"))
}

# update booster with dtrain
xgb.update <- function(booster, dtrain, iter) {
  if (class(booster) != "xgb.Booster") {
    stop("xgb.update: first argument must be type xgb.Booster")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.update: second argument must be type xgb.DMatrix")
  }
  .Call("XGBoosterUpdateOneIter_R", booster, as.integer(iter), dtrain)
  return(TRUE)
}
# evaluate one iteration
xgb.eval <- function(booster, watchlist, iter) {
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
xgb.save <- function(handle, fname) {
  if (typeof(fname) == "character") {
    stop("xgb.save: fname must be character");
  }
  if (class(handle) != "xgb.Booster") {
    .Call("XGBoosterSaveModel_R", handle, fname);
    return(TRUE)
  }
  if (class(handle) != "xgb.DMatrix") {
    
  }
}


# test code here
dtrain <- xgb.DMatrix("example/agaricus.txt.train")
dtest <- xgb.DMatrix("example/agaricus.txt.test")
param <- list("bst:min_child_weight" = 10,
              "objective" = "binary:logistic"
              )
bst<- xgb.Booster(list(dtrain, dtest), param )
success <- xgb.update(bst, dtrain, 0)
watchlist <- list('train'=dtrain,'test'=dtest)
cat(xgb.eval(bst, watchlist, 0))
cat("\n")

      
