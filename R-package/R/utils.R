#
# This file is for the low level reuseable utility functions
# that are not supposed to be visibe to a user.
#

#
# General helper utilities ----------------------------------------------------
#

# SQL-style NVL shortcut.
NVL <- function(x, val) {
  if (is.null(x))
    return(val)
  if (is.vector(x)) {
    x[is.na(x)] <- val
    return(x)
  }
  if (typeof(x) == 'closure')
    return(x)
  stop('x of unsupported for NVL type')
}


#
# Low-level functions for boosting --------------------------------------------
#

# Merges booster params with whatever is provided in ...
# plus runs some checks
check.params <- function(params, ...) {
  if (typeof(params) != "list") 
    stop("params must be a list")
  
  # in R interface, allow for '.' instead of '_' in parameter names
  names(params) <- gsub("\\.", "_", names(params))
  
  # merge parameters from the params and the dots-expansion
  dot.params <- list(...)
  names(dot.params) <- gsub("\\.", "_", names(dot.params))
  if (length(intersect(names(params), names(dot.params))) > 0)
    stop("Same parameters in 'params' and in the call are not allowed. Please check your 'params' list.")
  params <- c(params, dot.params)
  
  # only multiple eval_metric's make sense
  name.freqs <- table(names(params))
  multi.names <- setdiff( names(name.freqs[name.freqs>1]), 'eval_metric')
  if (length(multi.names) > 0) {
    warning("The following parameters (other than 'eval_metric') were provided multiple times:\n\t",
            paste(multi.names, collapse=', '), "\n  Only the last value for each of them will be used.\n")
    # While xgboost itself would choose the last value for a multi-parameter, 
    # will do some clean-up here b/c multi-parameters could be used further in R code, and R would 
    # pick the 1st (not the last) value when multiple elements with the same name are present in a list.
    for (n in multi.names) {
      del.idx <- which(n == names(params))
      del.idx <- del.idx[-length(del.idx)]
      params[[del.idx]] <- NULL
    }
  }
  
  # for multiclass, expect num_class to be set
  if (typeof(params[['objective']]) == "character" &&
    substr(NVL(params[['objective']], 'x'), 1, 6) == 'multi:') {
    if (as.numeric(NVL(params[['num_class']], 0)) < 2)
      stop("'num_class' > 1 parameter must be set for multiclass classification")
  }
  
  return(params)
}


# Performs some checks related to custom objective function.
# WARNING: has side-effects and can modify 'params' and 'obj' in its calling frame
check.custom.obj <- function(env = parent.frame()) {
  if (!is.null(env$params[['objective']]) && !is.null(env$obj))
    stop("Setting objectives in 'params' and 'obj' at the same time is not allowed")
  
  if (!is.null(env$obj) && typeof(env$obj) != 'closure')
    stop("'obj' must be a function")
  
  # handle the case when custom objective function was provided through params
  if (!is.null(env$params[['objective']]) &&
      typeof(env$params$objective) == 'closure') {
    env$obj <- env$params$objective
    p <- env$params
    p$objective <- NULL
    env$params <- p
  }
}

# Performs some checks related to custom evaluation function.
# WARNING: has side-effects and can modify 'params' and 'feval' in its calling frame
check.custom.eval <- function(env = parent.frame()) {
  if (!is.null(env$params[['eval_metric']]) && !is.null(env$feval))
    stop("Setting evaluation metrics in 'params' and 'feval' at the same time is not allowed")
  
  if (!is.null(env$feval) && typeof(env$feval) != 'closure')
    stop("'feval' must be a function")
  
  if (!is.null(env$feval) && is.null(env$maximize))
    stop("Please set 'maximize' to indicate whether the metric needs to be maximized or not")
  
  # handle the situation when custom eval function was provided through params
  if (!is.null(env$params[['eval_metric']]) &&
      typeof(env$params$eval_metric) == 'closure') {
    env$feval <- env$params$eval_metric
    p <- env$params
    p[ which(names(p) == 'eval_metric') ] <- NULL
    env$params <- p
  }
}


# Update booster with dtrain for an iteration
xgb.iter.update <- function(booster, dtrain, iter, obj = NULL) {
  if (class(booster) != "xgb.Booster.handle") {
    stop("first argument type must be xgb.Booster.handle")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("second argument type must be xgb.DMatrix")
  }

  if (is.null(obj)) {
    .Call("XGBoosterUpdateOneIter_R", booster, as.integer(iter), dtrain,
          PACKAGE = "xgboost")
  } else {
    pred <- predict(booster, dtrain)
    gpair <- obj(pred, dtrain)
    .Call("XGBoosterBoostOneIter_R", booster, dtrain, gpair$grad, gpair$hess, PACKAGE = "xgboost")
  }
  return(TRUE)
}


# Evaluate one iteration.
# Returns a named vector of evaluation metrics 
# with the names in a 'datasetname-metricname' format.
xgb.iter.eval <- function(booster, watchlist, iter, feval = NULL) {
  if (class(booster) != "xgb.Booster.handle")
    stop("first argument must be type xgb.Booster.handle")
  if (length(watchlist) == 0) 
    return(NULL)
  
  evnames <- names(watchlist)
  if (is.null(feval)) {
    msg <- .Call("XGBoosterEvalOneIter_R", booster, as.integer(iter), watchlist,
                 as.list(evnames), PACKAGE = "xgboost")
    msg <- str_split(msg, '(\\s+|:|\\s+)')[[1]][-1]
    res <- as.numeric(msg[c(FALSE,TRUE)]) # even indices are the values
    names(res) <- msg[c(TRUE,FALSE)]      # odds are the names
  } else {
    res <- sapply(seq_along(watchlist), function(j) {
      w <- watchlist[[j]]
      preds <- predict(booster, w) # predict using all trees
      eval_res <- feval(preds, w)
      out <- eval_res$value
      names(out) <- paste0(evnames[j], "-", eval_res$metric)
      out
    })
  }
  return(res)
}


#
# Helper functions for cross validation ---------------------------------------
#

# Generates random (stratified if needed) CV folds
generate.cv.folds <- function(nfold, nrows, stratified, label, params) {
  
  # cannot do it for rank
  if (exists('objective', where=params) &&
      is.character(params$objective) &&
      strtrim(params$objective, 5) == 'rank:')
    stop("\n\tAutomatic generation of CV-folds is not implemented for ranking!\n",
         "\tConsider providing pre-computed CV-folds through the 'folds=' parameter.\n")
  
  # shuffle
  rnd.idx <- sample(1:nrows)
  if (stratified &&
      length(label) == length(rnd.idx)) {
    y <- label[rnd.idx]
    # WARNING: some heuristic logic is employed to identify classification setting!
    #  - For classification, need to convert y labels to factor before making the folds,
    #    and then do stratification by factor levels.
    #  - For regression, leave y numeric and do stratification by quantiles.
    if (exists('objective', where=params) &&
        is.character(params$objective)) {
      # If 'objective' provided in params, assume that y is a classification label
      # unless objective is reg:linear
      if (params$objective != 'reg:linear')
        y <- factor(y)
    } else {
      # If no 'objective' given in params, it means that user either wants to use
      # the default 'reg:linear' objective or has provided a custom obj function.
      # Here, assume classification setting when y has 5 or less unique values:
      if (length(unique(y)) <= 5)
        y <- factor(y)
    }
    folds <- xgb.createFolds(y, nfold)
  } else {
    # make simple non-stratified folds
    kstep <- length(rnd.idx) %/% nfold
    folds <- list()
    for (i in 1:(nfold - 1)) {
      folds[[i]] <- rnd.idx[1:kstep]
      rnd.idx <- rnd.idx[-(1:kstep)]
    }
    folds[[nfold]] <- rnd.idx
  }
  return(folds)
}

# Creates CV folds stratified by the values of y.
# It was borrowed from caret::createFolds and simplified
# by always returning an unnamed list of fold indices.
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
