#' @importClassesFrom Matrix dgCMatrix dgeMatrix
#' @import methods

# depends on matrix
.onLoad <- function(libname, pkgname) {
  library.dynam("xgboost", pkgname, libname)
}
.onUnload <- function(libpath) {
  library.dynam.unload("xgboost", libpath)
}


## ----the following are low level iterative functions, not needed if
## you do not want to use them ---------------------------------------

# iteratively update booster with customized statistics
xgb.iter.boost <- function(booster, dtrain, gpair) {
  if (class(booster) != "xgb.Booster.handle") {
    stop("xgb.iter.update: first argument must be type xgb.Booster.handle")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.iter.update: second argument must be type xgb.DMatrix")
  }
  .Call("XGBoosterBoostOneIter_R", booster, dtrain, gpair$grad, gpair$hess, PACKAGE = "xgboost")
  return(TRUE)
}

# iteratively update booster with dtrain
xgb.iter.update <- function(booster, dtrain, iter, obj = NULL) {
  if (class(booster) != "xgb.Booster.handle") {
    stop("xgb.iter.update: first argument must be type xgb.Booster.handle")
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
xgb.iter.eval <- function(booster, watchlist, iter, feval = NULL, prediction = FALSE) {
  if (class(booster) != "xgb.Booster.handle") {
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
        preds <- predict(booster, w[[1]])
        ret <- feval(preds, w[[1]])
        msg <- paste(msg, "\t", names(w), "-", ret$metric, ":", ret$value, sep="")
      }
    }
  } else {
    msg <- ""
  }
  if (prediction){
    preds <- predict(booster,watchlist[[2]])
    return(list(msg,preds))
  }
  return(msg)
}

#------------------------------------------
# helper functions for cross validation
#
xgb.cv.mknfold <- function(dall, nfold, param, stratified, folds) {
  if (nfold <= 1) {
    stop("nfold must be bigger than 1")
  }
  if(is.null(folds)) {
    if (exists('objective', where=param) && is.character(param$objective) &&
        strtrim(param[['objective']], 5) == 'rank:') {
      stop("\tAutomatic creation of CV-folds is not implemented for ranking!\n",
           "\tConsider providing pre-computed CV-folds through the folds parameter.")
    }
    y <- getinfo(dall, 'label')
    randidx <- sample(1 : nrow(dall))
    if (stratified & length(y) == length(randidx)) {
      y <- y[randidx]
      #
      # WARNING: some heuristic logic is employed to identify classification setting!
      #
      # For classification, need to convert y labels to factor before making the folds,
      # and then do stratification by factor levels.
      # For regression, leave y numeric and do stratification by quantiles.
      if (exists('objective', where=param) && is.character(param$objective)) {
        # If 'objective' provided in params, assume that y is a classification label
        # unless objective is reg:linear
        if (param[['objective']] != 'reg:linear') y <- factor(y)
      } else {
        # If no 'objective' given in params, it means that user either wants to use
        # the default 'reg:linear' objective or has provided a custom obj function.
        # Here, assume classification setting when y has 5 or less unique values:
        if (length(unique(y)) <= 5) y <- factor(y)
      }
      folds <- xgb.createFolds(y, nfold)
    } else {
      # make simple non-stratified folds
      kstep <- length(randidx) %/% nfold
      folds <- list()
      for (i in 1:(nfold - 1)) {
        folds[[i]] <- randidx[1:kstep]
        randidx <- setdiff(randidx, folds[[i]])
      }
      folds[[nfold]] <- randidx
    }
  }
  ret <- list()
  for (k in 1:nfold) {
    dtest <- slice(dall, folds[[k]])
    didx <- c()
    for (i in 1:nfold) {
      if (i != k) {
        didx <- append(didx, folds[[i]])
      }
    }
    dtrain <- slice(dall, didx)
    bst <- xgb.Booster(param, list(dtrain, dtest))
    watchlist <- list(train=dtrain, test=dtest)
    ret[[k]] <- list(dtrain=dtrain, booster=bst, watchlist=watchlist, index=folds[[k]])
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
      ret <- paste(ret, sprintf("+%f", stats::sd(stats)), sep="")
    }
  }
  return (ret)
}

# Shamelessly copied from caret::createFolds
# and simplified by always returning an unnamed list of test indices
xgb.createFolds <- function(y, k = 10)
{
  if(is.numeric(y)) {
    ## Group the numeric data based on their magnitudes
    ## and sample within those groups.

    ## When the number of samples is low, we may have
    ## issues further slicing the numeric data into
    ## groups. The number of groups will depend on the
    ## ratio of the number of folds to the sample size.
    ## At most, we will use quantiles. If the sample
    ## is too small, we just do regular unstratified
    ## CV
    cuts <- floor(length(y) / k)
    if (cuts < 2) cuts <- 2
    if (cuts > 5) cuts <- 5
    y <- cut(y,
             unique(stats::quantile(y, probs = seq(0, 1, length = cuts))),
             include.lowest = TRUE)
  }

  if(k < length(y)) {
    ## reset levels so that the possible levels and
    ## the levels in the vector are the same
    y <- factor(as.character(y))
    numInClass <- table(y)
    foldVector <- vector(mode = "integer", length(y))

    ## For each class, balance the fold allocation as far
    ## as possible, then resample the remainder.
    ## The final assignment of folds is also randomized.
    for(i in 1:length(numInClass)) {
      ## create a vector of integers from 1:k as many times as possible without
      ## going over the number of samples in the class. Note that if the number
      ## of samples in a class is less than k, nothing is producd here.
      seqVector <- rep(1:k, numInClass[i] %/% k)
      ## add enough random integers to get  length(seqVector) == numInClass[i]
      if(numInClass[i] %% k > 0) seqVector <- c(seqVector, sample(1:k, numInClass[i] %% k))
      ## shuffle the integers for fold assignment and assign to this classes's data
      foldVector[which(y == dimnames(numInClass)$y[i])] <- sample(seqVector)
    }
  } else foldVector <- seq(along = y)

  out <- split(seq(along = y), foldVector)
  names(out) <- NULL
  out
}
