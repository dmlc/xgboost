#' Cross Validation
#' 
#' The cross validation function of xgboost
#' 
#' @param params the list of parameters. Commonly used ones are:
#' \itemize{
#'   \item \code{objective} objective function, common ones are
#'   \itemize{
#'     \item \code{reg:linear} linear regression
#'     \item \code{binary:logistic} logistic regression for classification
#'   }
#'   \item \code{eta} step size of each boosting step
#'   \item \code{max_depth} maximum depth of the tree
#'   \item \code{nthread} number of thread used in training, if not set, all threads are used
#' }
#'
#'   See \code{\link{xgb.train}} for further details.
#'   See also demo/ for walkthrough example in R.
#' @param data takes an \code{xgb.DMatrix}, \code{matrix}, or \code{dgCMatrix} as the input.
#' @param nrounds the max number of iterations
#' @param nfold the original dataset is randomly partitioned into \code{nfold} equal size subsamples. 
#' @param label vector of response values. Should be provided only when data is an R-matrix.
#' @param missing is only used when input is a dense matrix. By default is set to NA, which means 
#'        that NA values should be considered as 'missing' by the algorithm. 
#'        Sometimes, 0 or other extreme value might be used to represent missing values.
#' @param prediction A logical value indicating whether to return the test fold predictions 
#'        from each CV model. This parameter engages the \code{\link{cb.cv.predict}} callback.
#' @param showsd \code{boolean}, whether to show standard deviation of cross validation
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
#'        gradient with given prediction and dtrain.
#' @param feval custimized evaluation function. Returns 
#'        \code{list(metric='metric-name', value='metric-value')} with given 
#'        prediction and dtrain.
#' @param stratified a \code{boolean} indicating whether sampling of folds should be stratified 
#'        by the values of outcome labels.
#' @param folds \code{list} provides a possibility to use a list of pre-defined CV folds
#'        (each element must be a vector of test fold's indices). When folds are supplied, 
#'        the \code{nfold} and \code{stratified} parameters are ignored.
#' @param verbose \code{boolean}, print the statistics during the process
#' @param print_every_n Print each n-th iteration evaluation messages when \code{verbose>0}.
#'        Default is 1 which means all messages are printed. This parameter is passed to the 
#'        \code{\link{cb.print.evaluation}} callback.
#' @param early_stopping_rounds If \code{NULL}, the early stopping function is not triggered. 
#'        If set to an integer \code{k}, training with a validation set will stop if the performance 
#'        doesn't improve for \code{k} rounds.
#'        Setting this parameter engages the \code{\link{cb.early.stop}} callback.
#' @param maximize If \code{feval} and \code{early_stopping_rounds} are set,
#'        then this parameter must be set as well.
#'        When it is \code{TRUE}, it means the larger the evaluation score the better.
#'        This parameter is passed to the \code{\link{cb.early.stop}} callback.
#' @param callbacks a list of callback functions to perform various task during boosting.
#'        See \code{\link{callbacks}}. Some of the callbacks are automatically created depending on the 
#'        parameters' values. User can provide either existing or their own callback methods in order 
#'        to customize the training process.
#' @param ... other parameters to pass to \code{params}.
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
#' @return 
#' An object of class \code{xgb.cv.synchronous} with the following elements:
#' \itemize{
#'   \item \code{call} a function call.
#'   \item \code{params} parameters that were passed to the xgboost library. Note that it does not 
#'         capture parameters changed by the \code{\link{cb.reset.parameters}} callback.
#'   \item \code{callbacks} callback functions that were either automatically assigned or 
#'         explicitely passed.
#'   \item \code{evaluation_log} evaluation history storead as a \code{data.table} with the
#'         first column corresponding to iteration number and the rest corresponding to the 
#'         CV-based evaluation means and standard deviations for the training and test CV-sets.
#'         It is created by the \code{\link{cb.evaluation.log}} callback.
#'   \item \code{niter} number of boosting iterations.
#'   \item \code{folds} the list of CV folds' indices - either those passed through the \code{folds} 
#'         parameter or randomly generated.
#'   \item \code{best_iteration} iteration number with the best evaluation metric value
#'         (only available with early stopping).
#'   \item \code{best_ntreelimit} the \code{ntreelimit} value corresponding to the best iteration, 
#'         which could further be used in \code{predict} method
#'         (only available with early stopping).
#'   \item \code{pred} CV prediction values available when \code{prediction} is set. 
#'         It is either vector or matrix (see \code{\link{cb.cv.predict}}).
#'   \item \code{models} a liost of the CV folds' models. It is only available with the explicit 
#'         setting of the \code{cb.cv.predict(save_models = TRUE)} callback.
#' }
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
#' cv <- xgb.cv(data = dtrain, nrounds = 3, nthread = 2, nfold = 5, metrics = list("rmse","auc"),
#'                   max_depth = 3, eta = 1, objective = "binary:logistic")
#' print(cv)
#' print(cv, verbose=TRUE)
#' 
#' @export
xgb.cv <- function(params=list(), data, nrounds, nfold, label = NULL, missing = NA,
                   prediction = FALSE, showsd = TRUE, metrics=list(),
                   obj = NULL, feval = NULL, stratified = TRUE, folds = NULL, 
                   verbose = TRUE, print_every_n=1L,
                   early_stopping_rounds = NULL, maximize = NULL, callbacks = list(), ...) {

  check.deprecation(...)
  
  params <- check.booster.params(params, ...)
  # TODO: should we deprecate the redundant 'metrics' parameter?
  for (m in metrics)
    params <- c(params, list("eval_metric" = m))
  
  check.custom.obj()
  check.custom.eval()

  #if (is.null(params[['eval_metric']]) && is.null(feval))
  #  stop("Either 'eval_metric' or 'feval' must be provided for CV")
  
  # Check the labels
  if ( (inherits(data, 'xgb.DMatrix') && is.null(getinfo(data, 'label'))) ||
       (!inherits(data, 'xgb.DMatrix') && is.null(label)))
    stop("Labels must be provided for CV either through xgb.DMatrix, or through 'label=' when 'data' is matrix")
  
  # CV folds
  if(!is.null(folds)) {
    if(!is.list(folds) || length(folds) < 2)
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
  print_every_n <- max( as.integer(print_every_n), 1L)
  if (!has.callbacks(callbacks, 'cb.print.evaluation') && verbose) {
    callbacks <- add.cb(callbacks, cb.print.evaluation(print_every_n, showsd = showsd))
  }
  # evaluation log callback: always is on in CV
  evaluation_log <- list()
  if (!has.callbacks(callbacks, 'cb.evaluation.log')) {
    callbacks <- add.cb(callbacks, cb.evaluation.log())
  }
  # Early stopping callback
  stop_condition <- FALSE
  if (!is.null(early_stopping_rounds) &&
      !has.callbacks(callbacks, 'cb.early.stop')) {
    callbacks <- add.cb(callbacks, cb.early.stop(early_stopping_rounds, 
                                                 maximize = maximize, verbose = verbose))
  }
  # CV-predictions callback
  if (prediction &&
      !has.callbacks(callbacks, 'cb.cv.predict')) {
    callbacks <- add.cb(callbacks, cb.cv.predict(save_models = FALSE))
  }
  # Sort the callbacks into categories
  cb <- categorize.callbacks(callbacks)

  
  # create the booster-folds
  dall <- xgb.get.DMatrix(data, label, missing)
  bst_folds <- lapply(seq_along(folds), function(k) {
    dtest  <- slice(dall, folds[[k]])
    dtrain <- slice(dall, unlist(folds[-k]))
    handle <- xgb.Booster.handle(params, list(dtrain, dtest))
    list(dtrain = dtrain, bst = handle, watchlist = list(train = dtrain, test=dtest), index = folds[[k]])
  })
  # a "basket" to collect some results from callbacks
  basket <- list()

  # extract parameters that can affect the relationship b/w #trees and #iterations
  num_class <- max(as.numeric(NVL(params[['num_class']], 1)), 1)
  num_parallel_tree <- max(as.numeric(NVL(params[['num_parallel_tree']], 1)), 1)

  # those are fixed for CV (no training continuation)
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
  for (f in cb$finalize) f(finalize = TRUE)

  # the CV result
  ret <- list(
    call = match.call(),
    params = params,
    callbacks = callbacks,
    evaluation_log = evaluation_log,
    niter = end_iteration,
    folds = folds
  )
  ret <- c(ret, basket)

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
#' cv <- xgb.cv(data = train$data, label = train$label, nfold = 5, max_depth = 2,
#'                eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
#' print(cv)
#' print(cv, verbose=TRUE)
#' 
#' @rdname print.xgb.cv
#' @method print xgb.cv.synchronous
#' @export
print.xgb.cv.synchronous <- function(x, verbose = FALSE, ...) {
  cat('##### xgb.cv ', length(x$folds), '-folds\n', sep = '')
  
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
                 sep = ' = ', collapse = ', '), '\n', sep = '')
    }
    if (!is.null(x$callbacks) && length(x$callbacks) > 0) {
      cat('callbacks:\n')
      lapply(callback.calls(x$callbacks), function(x) {
        cat('  ')
        print(x)
      })
    }
    
    for (n in c('niter', 'best_iteration', 'best_ntreelimit')) {
      if (is.null(x[[n]])) 
        next
      cat(n, ': ', x[[n]], '\n', sep = '')
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
