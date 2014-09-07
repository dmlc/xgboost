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
      .Call("XGBoosterSetParam_R", handle, gsub("\\.", "_", names(p)), as.character(p),
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

## ----the following are low level iteratively function, not needed if
## you do not want to use them ---------------------------------------
# get dmatrix from data, label
xgb.get.DMatrix <- function(data, label = NULL) {
  inClass <- class(data)
  if (inClass == "dgCMatrix" || inClass == "matrix") {
    if (is.null(label)) {
      stop("xgboost: need label when data is a matrix")
    }
    dtrain <- xgb.DMatrix(data, label = label)
  } else {
    if (!is.null(label)) {
      warning("xgboost: label will be ignored.")
    }
    if (inClass == "character") {
      dtrain <- xgb.DMatrix(data)
    } else if (inClass == "xgb.DMatrix") {
      dtrain <- data
    } else {
      stop("xgboost: Invalid input of data")
    }
  }
  return (dtrain)
}
xgb.numrow <- function(dmat) {
  nrow <- .Call("XGDMatrixNumRow_R", dmat, PACKAGE="xgboost")
  return(nrow)
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

# iteratively update booster with dtrain
xgb.iter.update <- function(booster, dtrain, iter, obj = NULL) {
  if (class(booster) != "xgb.Booster") {
    stop("xgb.iter.update: first argument must be type xgb.Booster")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.iter.update: second argument must be type xgb.DMatrix")
  }

  if (is.null(obj)) {
    .Call("XGBoosterUpdateOneIter_R", booster, as.integer(iter), dtrain, 
          PACKAGE = "xgboost")
  } else {
    pred <- predict(booster, dtrain)
    gpair <- obj(pred, dtrain)
    succ <- xgb.iter.boost(booster, dtrain, gpair)
  }
  return(TRUE)
}

# iteratively evaluate one iteration
xgb.iter.eval <- function(booster, watchlist, iter, feval = NULL) {
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
  if (length(watchlist) != 0) {
    if (is.null(feval)) {
      evnames <- list()
      for (i in 1:length(watchlist)) {
        w <- watchlist[i]
        if (length(names(w)) == 0) {
          stop("xgb.eval: name tag must be presented for every elements in watchlist")
        }
        evnames <- append(evnames, names(w))
      }
      msg <- .Call("XGBoosterEvalOneIter_R", booster, as.integer(iter), watchlist, 
                   evnames, PACKAGE = "xgboost")
    } else {
      msg <- paste("[", iter, "]", sep="")
      for (j in 1:length(watchlist)) {
        w <- watchlist[j]
        if (length(names(w)) == 0) {
          stop("xgb.eval: name tag must be presented for every elements in watchlist")
        }
        ret <- feval(predict(booster, w[[1]]), w[[1]])
        msg <- paste(msg, "\t", names(w), "-", ret$metric, ":", ret$value, sep="")
      }
    }
  } else {
    msg <- ""
  }
  return(msg)
} 
#------------------------------------------
# helper functions for cross validation
#
xgb.cv.mknfold <- function(dall, nfold, param) {
  randidx <- sample(1 : xgb.numrow(dall))
  kstep <- length(randidx) / nfold
  idset <- list()
  for (i in 1:nfold) {
    idset[[i]] <- randidx[ ((i-1) * kstep + 1) : min(i * kstep, length(randidx)) ]
  }
  ret <- list()
  for (k in 1:nfold) {
    dtest <- slice(dall, idset[[k]])
    didx = c()
    for (i in 1:nfold) {
      if (i != k) {
        didx <- append(didx, idset[[i]])
      }
    }
    dtrain <- slice(dall, didx)
    bst <- xgb.Booster(param, list(dtrain, dtest))
    watchlist = list(train=dtrain, test=dtest)
    ret[[k]] <- list(dtrain=dtrain, booster=bst, watchlist=watchlist)
  }
  return (ret)
}
xgb.cv.aggcv <- function(res, showsd = TRUE) {
  header <- res[[1]]
  ret <- header[1]
  for (i in 2:length(header)) {
    kv <- strsplit(header[i], ":")[[1]]
    ret <- paste(ret, "\t", kv[1], ":", sep="")
    stats <- c()
    stats[1] <- as.numeric(kv[2])    
    for (j in 2:length(res)) {
      tkv <- strsplit(res[[j]][i], ":")[[1]]
      stats[j] <- as.numeric(tkv[2])
    }
    ret <- paste(ret, sprintf("%f", mean(stats)), sep="")
    if (showsd) {
      ret <- paste(ret, sprintf("+%f", sd(stats)), sep="")
    }
  }
  return (ret)
}
