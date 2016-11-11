# Construct a Booster from cachelist
# internal utility function
xgb.Booster <- function(params = list(), cachelist = list(), modelfile = NULL) {
  if (typeof(cachelist) != "list" ||
      any(sapply(cachelist, class) != 'xgb.DMatrix')) {
    stop("xgb.Booster only accepts list of DMatrix as cachelist")
  }

  handle <- .Call("XGBoosterCreate_R", cachelist, PACKAGE = "xgboost")
  if (!is.null(modelfile)) {
    if (typeof(modelfile) == "character") {
      .Call("XGBoosterLoadModel_R", handle, modelfile, PACKAGE = "xgboost")
    } else if (typeof(modelfile) == "raw") {
      .Call("XGBoosterLoadModelFromRaw_R", handle, modelfile, PACKAGE = "xgboost")
    } else if (class(modelfile) == "xgb.Booster") {
      modelfile <- xgb.Booster.check(modelfile, saveraw=TRUE)
      .Call("XGBoosterLoadModelFromRaw_R", handle, modelfile$raw, PACKAGE = "xgboost")
    } else {
      stop("modelfile must be either character filename, or raw booster dump, or xgb.Booster object")
    }
  }
  class(handle) <- "xgb.Booster.handle"
  if (length(params) > 0) {
    xgb.parameters(handle) <- params
  }
  return(handle)
}

# Convert xgb.Booster.handle to xgb.Booster
# internal utility function
xgb.handleToBooster <- function(handle, raw = NULL) {
  bst <- list(handle = handle, raw = raw)
  class(bst) <- "xgb.Booster"
  return(bst)
}

# Return a verified to be valid handle out of either xgb.Booster.handle or xgb.Booster
# internal utility function
xgb.get.handle <- function(object) {
  handle <- switch(class(object)[1],
    xgb.Booster = object$handle,
    xgb.Booster.handle = object,
    stop("argument must be of either xgb.Booster or xgb.Booster.handle class")
  )
  if (is.null(handle) || .Call("XGCheckNullPtr_R", handle, PACKAGE="xgboost")) {
    stop("invalid xgb.Booster.handle")
  }
  handle
}

# Check whether an xgb.Booster object is complete
# internal utility function
xgb.Booster.check <- function(bst, saveraw = TRUE) {
  if (class(bst) != "xgb.Booster")
    stop("argument type must be xgb.Booster")
  
  isnull <- is.null(bst$handle)
  if (!isnull) {
    isnull <- .Call("XGCheckNullPtr_R", bst$handle, PACKAGE="xgboost")
  }
  if (isnull) {
    bst$handle <- xgb.Booster(modelfile = bst$raw)
  } else {
    if (is.null(bst$raw) && saveraw)
      bst$raw <- xgb.save.raw(bst$handle)
  }
  return(bst)
}


#' Predict method for eXtreme Gradient Boosting model
#' 
#' Predicted values based on either xgboost model or model handle object.
#' 
#' @param object Object of class \code{xgb.Booster} or \code{xgb.Booster.handle}
#' @param newdata takes \code{matrix}, \code{dgCMatrix}, local data file or \code{xgb.DMatrix}.
#' @param missing Missing is only used when input is dense matrix. Pick a float value that represents
#'        missing values in data (e.g., sometimes 0 or some other extreme value is used).
#' @param outputmargin whether the prediction should be returned in the for of original untransformed 
#'        sum of predictions from boosting iterations' results. E.g., setting \code{outputmargin=TRUE} for 
#'        logistic regression would result in predictions for log-odds instead of probabilities.
#' @param ntreelimit limit the number of model's trees or boosting iterations used in prediction (see Details).
#'        It will use all the trees by default (\code{NULL} value).
#' @param predleaf whether predict leaf index instead. 
#' @param reshape whether to reshape the vector of predictions to a matrix form when there are several 
#'        prediction outputs per case. This option has no effect when \code{predleaf = TRUE}.
#' @param ... Parameters passed to \code{predict.xgb.Booster}
#' 
#' @details  
#' Note that \code{ntreelimit} is not necesserily equal to the number of boosting iterations
#' and it is not necesserily equal to the number of trees in a model.
#' E.g., in a random forest-like model, \code{ntreelimit} would limit the number of trees.
#' But for multiclass classification, there are multiple trees per iteration, 
#' but \code{ntreelimit} limits the number of boosting iterations.
#' 
#' Also note that \code{ntreelimit} would currently do nothing for predictions from gblinear, 
#' since gblinear doesn't keep its boosting history. 
#' 
#' One possible practical applications of the \code{predleaf} option is to use the model 
#' as a generator of new features which capture non-linearity and interactions, 
#' e.g., as implemented in \code{\link{xgb.create.features}}. 
#' 
#' @return 
#' For regression or binary classification, it returns a vector of length \code{nrows(newdata)}.
#' For multiclass classification, either a \code{num_class * nrows(newdata)} vector or 
#' a \code{(nrows(newdata), num_class)} dimension matrix is returned, depending on 
#' the \code{reshape} value.
#' 
#' When \code{predleaf = TRUE}, the output is a matrix object with the 
#' number of columns corresponding to the number of trees.
#' 
#' @seealso
#' \code{\link{xgb.train}}.
#' 
#' @examples
#' ## binary classification:
#' 
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' 
#' bst <- xgboost(data = train$data, label = train$label, max_depth = 2, 
#'                eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
#' # use all trees by default
#' pred <- predict(bst, test$data)
#' # use only the 1st tree
#' pred <- predict(bst, test$data, ntreelimit = 1)
#' 
#' 
#' ## multiclass classification in iris dataset:
#' 
#' lb <- as.numeric(iris$Species) - 1
#' num_class <- 3
#' set.seed(11)
#' bst <- xgboost(data = as.matrix(iris[, -5]), label = lb,
#'                max_depth = 4, eta = 0.5, nthread = 2, nrounds = 10, subsample = 0.5,
#'                objective = "multi:softprob", num_class = num_class)
#' # predict for softmax returns num_class probability numbers per case:
#' pred <- predict(bst, as.matrix(iris[, -5]))
#' str(pred)
#' # reshape it to a num_class-columns matrix
#' pred <- matrix(pred, ncol=num_class, byrow=TRUE)
#' # convert the probabilities to softmax labels
#' pred_labels <- max.col(pred) - 1
#' # the following should result in the same error as seen in the last iteration
#' sum(pred_labels != lb)/length(lb)
#' 
#' # compare that to the predictions from softmax:
#' set.seed(11)
#' bst <- xgboost(data = as.matrix(iris[, -5]), label = lb,
#'                max_depth = 4, eta = 0.5, nthread = 2, nrounds = 10, subsample = 0.5,
#'                objective = "multi:softmax", num_class = num_class)
#' pred <- predict(bst, as.matrix(iris[, -5]))
#' str(pred)
#' all.equal(pred, pred_labels)
#' # prediction from using only 5 iterations should result 
#' # in the same error as seen in iteration 5:
#' pred5 <- predict(bst, as.matrix(iris[, -5]), ntreelimit=5)
#' sum(pred5 != lb)/length(lb)
#' 
#' 
#' ## random forest-like model of 25 trees for binary classification:
#' 
#' set.seed(11)
#' bst <- xgboost(data = train$data, label = train$label, max_depth = 5,
#'                nthread = 2, nrounds = 1, objective = "binary:logistic",
#'                num_parallel_tree = 25, subsample = 0.6, colsample_bytree = 0.1)
#' # Inspect the prediction error vs number of trees:
#' lb <- test$label
#' dtest <- xgb.DMatrix(test$data, label=lb)
#' err <- sapply(1:25, function(n) {
#'   pred <- predict(bst, dtest, ntreelimit=n)
#'   sum((pred > 0.5) != lb)/length(lb)
#' })
#' plot(err, type='l', ylim=c(0,0.1), xlab='#trees')
#'
#' @rdname predict.xgb.Booster
#' @export
predict.xgb.Booster <- function(object, newdata = NULL, missing = NA,
    outputmargin = FALSE, ntreelimit = NULL, predleaf = FALSE, reshape = FALSE, ...) {

  object <- xgb.Booster.check(object, saveraw = FALSE)
  if(is.null(newdata)){
    newdata = object$data
  }
  if (class(newdata) != "xgb.DMatrix")
    newdata <- xgb.DMatrix(newdata, missing = missing)
  if (is.null(ntreelimit))
    ntreelimit <- NVL(object$best_ntreelimit, 0)
  if (ntreelimit < 0)
    stop("ntreelimit cannot be negative")
  
  option <- 0L + 1L * as.logical(outputmargin) + 2L * as.logical(predleaf)
  
  ret <- .Call("XGBoosterPredict_R", object$handle, newdata, option[1],
               as.integer(ntreelimit), PACKAGE = "xgboost")
  
  if (length(ret) %% nrow(newdata) != 0)
    stop("prediction length ", length(ret)," is not multiple of nrows(newdata) ", nrow(newdata))
  npred_per_case <- length(ret) / nrow(newdata)

  if (predleaf){
    len <- nrow(newdata)
    ret <- if (length(ret) == len) {
      matrix(ret, ncol = 1)
    } else {
      t(matrix(ret, ncol = len))
    }
  } else if (reshape && npred_per_case > 1) {
    ret <- matrix(ret, ncol = length(ret) / nrow(newdata), byrow = TRUE)
  }
  return(ret)
}

#' @rdname predict.xgb.Booster
#' @export
predict.xgb.Booster.handle <- function(object, ...) {

  bst <- xgb.handleToBooster(object)

  ret <- predict(bst, ...)
  return(ret)
}


#' Accessors for serializable attributes of a model.
#'
#' These methods allow to manipulate the key-value attribute strings of an xgboost model.
#'
#' @param object Object of class \code{xgb.Booster} or \code{xgb.Booster.handle}.
#' @param name a non-empty character string specifying which attribute is to be accessed.
#' @param value a value of an attribute for \code{xgb.attr<-}; for \code{xgb.attributes<-} 
#'        it's a list (or an object coercible to a list) with the names of attributes to set 
#'        and the elements corresponding to attribute values. 
#'        Non-character values are converted to character.
#'        When attribute value is not a scalar, only the first index is used.
#'        Use \code{NULL} to remove an attribute.
#'
#' @details
#' The primary purpose of xgboost model attributes is to store some meta-data about the model.
#' Note that they are a separate concept from the object attributes in R.
#' Specifically, they refer to key-value strings that can be attached to an xgboost model,
#' stored together with the model's binary representation, and accessed later 
#' (from R or any other interface).
#' In contrast, any R-attribute assigned to an R-object of \code{xgb.Booster} class
#' would not be saved by \code{xgb.save} because an xgboost model is an external memory object
#' and its serialization is handled extrnally.
#' Also, setting an attribute that has the same name as one of xgboost's parameters wouldn't 
#' change the value of that parameter for a model. 
#' Use \code{\link{xgb.parameters<-}} to set or change model parameters.
#' 
#' The attribute setters would usually work more efficiently for \code{xgb.Booster.handle}
#' than for \code{xgb.Booster}, since only just a handle (pointer) would need to be copied.
#' That would only matter if attributes need to be set many times.
#' Note, however, that when feeding a handle of an \code{xgb.Booster} object to the attribute setters,
#' the raw model cache of an \code{xgb.Booster} object would not be automatically updated, 
#' and it would be user's responsibility to call \code{xgb.save.raw} to update it.
#' 
#' The \code{xgb.attributes<-} setter either updates the existing or adds one or several attributes, 
#' but it doesn't delete the other existing attributes.
#' 
#' @return
#' \code{xgb.attr} returns either a string value of an attribute 
#' or \code{NULL} if an attribute wasn't stored in a model.
#' 
#' \code{xgb.attributes} returns a list of all attribute stored in a model 
#' or \code{NULL} if a model has no stored attributes.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#'
#' bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
#'                eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
#'
#' xgb.attr(bst, "my_attribute") <- "my attribute value"
#' print(xgb.attr(bst, "my_attribute"))
#' xgb.attributes(bst) <- list(a = 123, b = "abc")
#'
#' xgb.save(bst, 'xgb.model')
#' bst1 <- xgb.load('xgb.model')
#' print(xgb.attr(bst1, "my_attribute"))
#' print(xgb.attributes(bst1))
#' 
#' # deletion:
#' xgb.attr(bst1, "my_attribute") <- NULL
#' print(xgb.attributes(bst1))
#' xgb.attributes(bst1) <- list(a = NULL, b = NULL)
#' print(xgb.attributes(bst1))
#' 
#' @rdname xgb.attr
#' @export
xgb.attr <- function(object, name) {
  if (is.null(name) || nchar(as.character(name[1])) == 0) stop("invalid attribute name")
  handle <- xgb.get.handle(object)
  .Call("XGBoosterGetAttr_R", handle, as.character(name[1]), PACKAGE="xgboost")
}

#' @rdname xgb.attr
#' @export
`xgb.attr<-` <- function(object, name, value) {
  if (is.null(name) || nchar(as.character(name[1])) == 0) stop("invalid attribute name")
  handle <- xgb.get.handle(object)
  if (!is.null(value)) {
    # Coerce the elements to be scalar strings.
    # Q: should we warn user about non-scalar elements?
    if (is.numeric(value[1])) {
      value <- format(value[1], digits = 17)
    } else {
      value <- as.character(value[1])
    }
  }
  .Call("XGBoosterSetAttr_R", handle, as.character(name[1]), value, PACKAGE="xgboost")
  if (is(object, 'xgb.Booster') && !is.null(object$raw)) {
    object$raw <- xgb.save.raw(object$handle)
  }
  object
}

#' @rdname xgb.attr
#' @export
xgb.attributes <- function(object) {
  handle <- xgb.get.handle(object)
  attr_names <- .Call("XGBoosterGetAttrNames_R", handle, PACKAGE="xgboost")
  if (is.null(attr_names)) return(NULL)
  res <- lapply(attr_names, function(x) {
    .Call("XGBoosterGetAttr_R", handle, x, PACKAGE="xgboost")
  })
  names(res) <- attr_names
  res
}

#' @rdname xgb.attr
#' @export
`xgb.attributes<-` <- function(object, value) {
  a <- as.list(value)
  if (is.null(names(a)) || any(nchar(names(a)) == 0)) {
    stop("attribute names cannot be empty strings")
  }
  # Coerce the elements to be scalar strings.
  # Q: should we warn a user about non-scalar elements?
  a <- lapply(a, function(x) {
    if (is.null(x)) return(NULL)
    if (is.numeric(value[1])) {
      format(x[1], digits = 17)
    } else {
      as.character(x[1])
    }
  })
  handle <- xgb.get.handle(object)
  for (i in seq_along(a)) {
    .Call("XGBoosterSetAttr_R", handle, names(a[i]), a[[i]], PACKAGE="xgboost")
  }
  if (is(object, 'xgb.Booster') && !is.null(object$raw)) {
    object$raw <- xgb.save.raw(object$handle)
  }
  object
}

#' Accessors for model parameters.
#'
#' Only the setter for xgboost parameters is currently implemented.
#'
#' @param object Object of class \code{xgb.Booster} or \code{xgb.Booster.handle}.
#' @param value a list (or an object coercible to a list) with the names of parameters to set
#'        and the elements corresponding to parameter values.
#'
#' @details
#' Note that the setter would usually work more efficiently for \code{xgb.Booster.handle}
#' than for \code{xgb.Booster}, since only just a handle would need to be copied.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#'
#' bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
#'                eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
#'
#' xgb.parameters(bst) <- list(eta = 0.1)
#' 
#' @rdname xgb.parameters
#' @export
`xgb.parameters<-` <- function(object, value) {
  if (length(value) == 0) return(object)
  p <- as.list(value)
  if (is.null(names(p)) || any(nchar(names(p)) == 0)) {
    stop("parameter names cannot be empty strings")
  }
  names(p) <- gsub("\\.", "_", names(p))
  p <- lapply(p, function(x) as.character(x)[1])
  handle <- xgb.get.handle(object)
  for (i in seq_along(p)) {
    .Call("XGBoosterSetParam_R", handle, names(p[i]), p[[i]], PACKAGE = "xgboost")
  }
  if (is(object, 'xgb.Booster') && !is.null(object$raw)) {
    object$raw <- xgb.save.raw(object$handle)
  }
  object
}

# Extract # of trees in a model
# TODO: either add a getter to C-interface, or simply set an 'ntree' attribute after each iteration
# internal utility function
xgb.ntree <- function(bst) {
  length(grep('^booster', xgb.dump(bst)))
}


#' Print xgb.Booster
#' 
#' Print information about xgb.Booster.
#' 
#' @param x an xgb.Booster object
#' @param verbose whether to print detailed data (e.g., attribute values)
#' @param ... not currently used
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' bst <- xgboost(data = train$data, label = train$label, max_depth = 2,
#'                eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
#' attr(bst, 'myattr') <- 'memo'
#' 
#' print(bst)
#' print(bst, verbose=TRUE)
#'
#' @method print xgb.Booster 
#' @export
print.xgb.Booster <- function(x, verbose=FALSE, ...) {
  cat('##### xgb.Booster\n')
  
  if (is.null(x$handle) || .Call("XGCheckNullPtr_R", x$handle, PACKAGE="xgboost")) {
    cat("handle is invalid\n")
    return(x)
  }
  
  cat('raw: ')
  if (!is.null(x$raw)) {
    cat(format(object.size(x$raw), units="auto"), '\n')
  } else {
    cat('NULL\n')
  }
  if (!is.null(x$call)) {
    cat('call:\n  ')
    print(x$call)
  }
  
  if (!is.null(x$params)) {
    cat('params (as set within xgb.train):\n')
    cat( '  ', 
         paste(names(x$params),
               paste0('"', unlist(x$params), '"'),
               sep=' = ', collapse=', '), '\n', sep='')
  }
  # TODO: need an interface to access all the xgboosts parameters

  attrs <- xgb.attributes(x)
  if (length(attrs) > 0) {
    cat('xgb.attributes:\n')
    if (verbose) {
      cat( paste(paste0('  ',names(attrs)),
                 paste0('"', unlist(attrs), '"'),
                 sep=' = ', collapse='\n'), '\n', sep='')
    } else {
      cat('  ', paste(names(attrs), collapse=', '), '\n', sep='')
    }
  }
  
  if (!is.null(x$callbacks) && length(x$callbacks) > 0) {
    cat('callbacks:\n')
    lapply(callback.calls(x$callbacks), function(x) {
      cat('  ')
      print(x)
    })
  }
  
  cat('niter: ', x$niter, '\n', sep='')
  # TODO: uncomment when faster xgb.ntree is implemented
  #cat('ntree: ', xgb.ntree(x), '\n', sep='')
  
  for (n in setdiff(names(x), c('handle', 'raw', 'call', 'params', 'callbacks','evaluation_log','niter'))) {
    if (is.atomic(x[[n]])) {
      cat(n, ': ', x[[n]], '\n', sep='')
    } else {
      cat(n, ':\n\t', sep='')
      print(x[[n]])
    }
  }
  
  if (!is.null(x$evaluation_log)) {
    cat('evaluation_log:\n')
    print(x$evaluation_log, row.names = FALSE, topn = 2)
  }
  
  invisible(x)
}
