#' Cross Validation
#' 
#' The cross valudation function of xgboost
#' 
#' @param params the list of parameters. Commonly used ones are:
#' \itemize{
#'   \item \code{objective} objective function, common ones are
#'   \itemize{
#'     \item \code{reg:linear} linear regression
#'     \item \code{binary:logistic} logistic regression for classification
#'   }
#'   \item \code{eta} step size of each boosting step
#'   \item \code{max.depth} maximum depth of the tree
#'   \item \code{nthread} number of thread used in training, if not set, all threads are used
#' }
#'
#'   See \link{xgb.train} for further details.
#'   See also demo/ for walkthrough example in R.
#' @param data takes an \code{xgb.DMatrix} or \code{Matrix} as the input.
#' @param nrounds the max number of iterations
#' @param nfold the original dataset is randomly partitioned into \code{nfold} equal size subsamples. 
#' @param label option field, when data is \code{Matrix}
#' @param missing Missing is only used when input is dense matrix, pick a float
#'     value that represents missing value. Sometime a data use 0 or other extreme value to represents missing values.
#' @param prediction A logical value indicating whether to return the prediction vector.
#' @param showsd \code{boolean}, whether show standard deviation of cross validation
#' @param metrics, list of evaluation metrics to be used in cross validation,
#'   when it is not specified, the evaluation metric is chosen according to objective function.
#'   Possible options are:
#' \itemize{
#'   \item \code{error} binary classification error rate
#'   \item \code{rmse} Rooted mean square error
#'   \item \code{logloss} negative log-likelihood function
#'   \item \code{auc} Area under curve
#'   \item \code{merror} Exact matching error, used to evaluate multi-class classification
#' }
#' @param obj customized objective function. Returns gradient and second order 
#'   gradient with given prediction and dtrain.
#' @param feval custimized evaluation function. Returns 
#'   \code{list(metric='metric-name', value='metric-value')} with given 
#'   prediction and dtrain.
#' @param stratified \code{boolean} whether sampling of folds should be stratified by the values of labels in \code{data}
#' @param folds \code{list} provides a possibility of using a list of pre-defined CV folds (each element must be a vector of fold's indices).
#'   If folds are supplied, the nfold and stratified parameters would be ignored.
#' @param verbose \code{boolean}, print the statistics during the process
#' @param print.every.n Print every N progress messages when \code{verbose>0}. Default is 1 which means all messages are printed.
#' @param early.stop.round If \code{NULL}, the early stopping function is not triggered. 
#'        If set to an integer \code{k}, training with a validation set will stop if the performance 
#'        doesn't improve for \code{k} rounds.
#' @param maximize If \code{feval} and \code{early.stop.round} are set, then \code{maximize} must be set as well.
#'        \code{maximize=TRUE} means the larger the evaluation score the better.
#'     
#' @param ... other parameters to pass to \code{params}.
#' 
#' @return
#' TODO: update this...
#' 
#' If \code{prediction = TRUE}, a list with the following elements is returned:
#' \itemize{
#'   \item \code{dt} a \code{data.table} with each mean and standard deviation stat for training set and test set
#'   \item \code{pred} an array or matrix (for multiclass classification) with predictions for each CV-fold for the model having been trained on the data in all other folds.
#' }
#'
#' If \code{prediction = FALSE}, just a \code{data.table} with each mean and standard deviation stat for training set and test set is returned.
#'
#' @details 
#' The original sample is randomly partitioned into \code{nfold} equal size subsamples. 
#' 
#' Of the \code{nfold} subsamples, a single subsample is retained as the validation data for testing the model, and the remaining \code{nfold - 1} subsamples are used as training data. 
#' 
#' The cross-validation process is then repeated \code{nrounds} times, with each of the \code{nfold} subsamples used exactly once as the validation data.
#' 
#' All observations are used for both training and validation.
#' 
#' Adapted from \url{http://en.wikipedia.org/wiki/Cross-validation_\%28statistics\%29#k-fold_cross-validation}
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
#' history <- xgb.cv(data = dtrain, nround=3, nthread = 2, nfold = 5, metrics=list("rmse","auc"),
#'                   max.depth =3, eta = 1, objective = "binary:logistic")
#' print(history)
#' 
#' @export
xgb.cv <- function(params=list(), data, nrounds, nfold, label = NULL, missing = NA,
                   prediction = FALSE, showsd = TRUE, metrics=list(),
                   obj = NULL, feval = NULL, stratified = TRUE, folds = NULL, 
                   verbose = TRUE, print.every.n=1L,
                   early.stop.round = NULL, maximize = NULL, callbacks = list(), ...) {

  #strategy <- match.arg(strategy)
  
  params <- check.params(params, ...)
  # TODO: should we deprecate the redundant 'metrics' parameter?
  for (m in metrics)
    params <- c(params, list("eval_metric" = m))
  
  check.custom.obj()
  check.custom.eval()

  #if (is.null(params[['eval_metric']]) && is.null(feval))
  #  stop("Either 'eval_metric' or 'feval' must be provided for CV")
  
  # Labels
  if (class(data) == 'xgb.DMatrix')
    labels <- getinfo(data, 'label')
  if (is.null(labels))
    stop("Labels must be provided for CV either through xgb.DMatrix, or through 'label=' when 'data' is matrix")
  
  # CV folds
  if(!is.null(folds)) {
    if(class(folds) != "list" || length(folds) < 2)
      stop("'folds' must be a list with 2 or more elements that are vectors of indices for each CV-fold")
    nfold <- length(folds)
  } else {
    if (nfold <= 1)
      stop("'nfold' must be > 1")
    folds <- generate.cv.folds(nfold, nrow(data), stratified, label, params)
  }
  
  # Potential TODO: sequential CV
  #if (strategy == 'sequential')
  #  stop('Sequential CV strategy is not yet implemented')

  # verbosity & evaluation printing callback:
  params <- c(params, list(silent = 1))
  print.every.n <- max( as.integer(print.every.n), 1L)
  if (!has.callbacks(callbacks, 'cb.print_evaluation') && verbose)
    callbacks <- c(callbacks, cb.print_evaluation(print.every.n))

  # evaluation log callback: always is on in CV
  evaluation_log <- list()
  if (!has.callbacks(callbacks, 'cb.log_evaluation'))
    callbacks <- c(callbacks, cb.log_evaluation())
  
  # Early stopping callback
  stop_condition <- FALSE
  if (!is.null(early.stop.round) &&
      !has.callbacks(callbacks, 'cb.early_stop'))
    callbacks <- c(callbacks, cb.early_stop(early.stop.round, maximize=maximize, verbose=verbose))
  
  # Sort the callbacks into categories
  names(callbacks) <- callback.names(callbacks)
  cb <- categorize.callbacks(callbacks)

  
  # create the booster-folds
  dall <- xgb.get.DMatrix(data, label, missing)
  bst_folds <- lapply(1:length(folds), function(k) {
    dtest  <- slice(dall, folds[[k]])
    dtrain <- slice(dall, unlist(folds[-k]))
    bst <- xgb.Booster(params, list(dtrain, dtest))
    list(dtrain=dtrain, bst=bst, watchlist=list(train=dtrain, test=dtest), index=folds[[k]])
  })

  num_class <- max(as.numeric(NVL(params[['num_class']], 1)), 1)
  num_parallel_tree <- max(as.numeric(NVL(params[['num_parallel_tree']], 1)), 1)

  begin_iteration <- 1
  end_iteration <- nrounds
  
  # synchronous CV boosting: run CV folds' models within each iteration
  for (iteration in begin_iteration:end_iteration) {
    
    for (f in cb$pre_iter) f()
    
    msg <- lapply(bst_folds, function(fd) {
      xgb.iter.update(fd$bst, fd$dtrain, iteration - 1, obj)
      xgb.iter.eval(fd$bst, fd$watchlist, iteration - 1, feval)
    })
    msg <- simplify2array(msg)
    bst_evaluation <- rowMeans(msg)
    bst_evaluation_err <- sqrt(rowMeans(msg^2) - bst_evaluation^2)

    for (f in cb$post_iter) f()
    
    if (stop_condition) break
  }
  for (f in cb$finalize) f(finalize=TRUE)

  # the CV result
  ret <- list(
    call = match.call(),
    params = params,
    callbacks = callbacks,
    evaluation_log = evaluation_log,
    nboost = end_iteration,
    ntree = end_iteration * num_parallel_tree * num_class
    )
  if (!is.null(attr(bst_folds, 'best_iteration'))) {
    ret$best_iteration <- attr(bst_folds, 'best_iteration')
    ret$best_ntreelimit <- attr(bst_folds, 'best_ntreelimit')
  }
  ret$folds <- folds

  # TODO: should making prediction go 
  #  a. into a callback?
  #  b. return folds' models, and have a separate method for predictions?
  if (prediction) {
    ret$pred <- ifelse(num_class > 1, 
                       matrix(0, nrow(data), num_class), 
                       rep(0, nrow(data)))
    ntreelimit <- NVL(ret$best_ntreelimit, ret$ntree)
    for (fd in bst_folds) {
      pred <- predict(fd$bst, fd$watchlist[[2]], ntreelimit = ntreelimit)
      if (is.matrix(ret$pred))
        ret$pred[fd$index,] <- t(matrix(pred, num_class, length(fd$index)))
      else
        ret$pred[fd$index] <- pred
    }
    ret$bst <- lapply(bst_folds, function(x) {
      xgb.Booster.check(xgb.handleToBooster(x$bst), saveraw = TRUE)
    })
  }
  
  class(ret) <- 'xgb.cv.synchronous'
  invisible(ret)
}



#' Print xgb.cv result
#' 
#' Prints formatted results of \code{xgb.cv}.
#' 
#' @param x an \code{xgb.cv.synchronous} object
#' @param verbose whether to print detailed data
#' @param ... passed to \code{data.table.print}
#' 
#' @details
#' When not verbose, it would only print the evaluation results, 
#' including the best iteration (when available).
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' cv <- xgb.cv(data = train$data, label = train$label, nfold = 5, max.depth = 2,
#'                eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
#' print(cv)
#' print(cv, verbose=TRUE)
#' 
#' @rdname print.xgb.cv
#' @export
print.xgb.cv.synchronous <- function(x, verbose=FALSE, ...) {
  cat('##### xgb.cv ', length(x$folds), '-folds\n', sep='')
  
  if (verbose) {
    if (!is.null(x$call)) {
      cat('call:\n  ')
      print(x$call)
    }
    if (!is.null(x$params)) {
      cat('params (as set within xgb.cv):\n')
      cat( '  ', 
           paste(names(x$params), 
                 paste0('"', unlist(x$params), '"'),
                 sep=' = ', collapse=', '), '\n', sep='')
    }
    if (!is.null(x$callbacks) && length(x$callbacks) > 0) {
      cat('callbacks:\n')
      lapply(callback.calls(x$callbacks), function(x) {
        cat('  ')
        print(x)
      })
    }
    
    for (n in c('nboost', 'ntree', 'best_iteration', 'best_ntreelimit')) {
      if (is.null(x[[n]])) 
        next
      cat(n, ': ', x[[n]], '\n', sep='')
    }

    if (!is.null(x$pred)) {
      cat('pred:\n')
      str(x$pred)
    }
  }

  if (verbose) 
    cat('evaluation_log:\n')
  print(x$evaluation_log, row.names = FALSE, ...)
  if (!is.null(x$best_iteration)) {
    cat('Best iteration:\n')
    print(x$evaluation_log[x$best_iteration], row.names = FALSE, ...)
  }
  invisible(x)
}
