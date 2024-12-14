#' Cross Validation
#'
#' The cross validation function of xgboost.
#'
#' @inheritParams xgb.train
#' @param data An `xgb.DMatrix` object, with corresponding fields like `label` or bounds as required
#'   for model training by the objective.
#'
#'   Note that only the basic `xgb.DMatrix` class is supported - variants such as `xgb.QuantileDMatrix`
#'   or `xgb.ExtMemDMatrix` are not supported here.
#' @param nfold The original dataset is randomly partitioned into `nfold` equal size subsamples.
#' @param prediction A logical value indicating whether to return the test fold predictions
#'   from each CV model. This parameter engages the [xgb.cb.cv.predict()] callback.
#' @param showsd Logical value whether to show standard deviation of cross validation.
#' @param metrics List of evaluation metrics to be used in cross validation,
#'   when it is not specified, the evaluation metric is chosen according to objective function.
#'   Possible options are:
#'   - `error`: Binary classification error rate
#'   - `rmse`: Root mean square error
#'   - `logloss`: Negative log-likelihood function
#'   - `mae`: Mean absolute error
#'   - `mape`: Mean absolute percentage error
#'   - `auc`: Area under curve
#'   - `aucpr`: Area under PR curve
#'   - `merror`: Exact matching error used to evaluate multi-class classification
#' @param stratified Logical flag indicating whether sampling of folds should be stratified
#'   by the values of outcome labels. For real-valued labels in regression objectives,
#'   stratification will be done by discretizing the labels into up to 5 buckets beforehand.
#'
#'   If passing "auto", will be set to `TRUE` if the objective in `params` is a classification
#'   objective (from XGBoost's built-in objectives, doesn't apply to custom ones), and to
#'   `FALSE` otherwise.
#'
#'   This parameter is ignored when `data` has a `group` field - in such case, the splitting
#'   will be based on whole groups (note that this might make the folds have different sizes).
#'
#'   Value `TRUE` here is **not** supported for custom objectives.
#' @param folds List with pre-defined CV folds (each element must be a vector of test fold's indices).
#'   When folds are supplied, the `nfold` and `stratified` parameters are ignored.
#'
#'   If `data` has a `group` field and the objective requires this field, each fold (list element)
#'   must additionally have two attributes (retrievable through `attributes`) named `group_test`
#'   and `group_train`, which should hold the `group` to assign through [setinfo.xgb.DMatrix()] to
#'   the resulting DMatrices.
#' @param train_folds List specifying which indices to use for training. If `NULL`
#'   (the default) all indices not specified in `folds` will be used for training.
#'
#'   This is not supported when `data` has `group` field.
#' @param callbacks A list of callback functions to perform various task during boosting.
#'   See [xgb.Callback()]. Some of the callbacks are automatically created depending on the
#'   parameters' values. User can provide either existing or their own callback methods in order
#'   to customize the training process.
#' @details
#' The original sample is randomly partitioned into `nfold` equal size subsamples.
#'
#' Of the `nfold` subsamples, a single subsample is retained as the validation data for testing the model,
#' and the remaining `nfold - 1` subsamples are used as training data.
#'
#' The cross-validation process is then repeated `nrounds` times, with each of the
#' `nfold` subsamples used exactly once as the validation data.
#'
#' All observations are used for both training and validation.
#'
#' Adapted from \url{https://en.wikipedia.org/wiki/Cross-validation_\%28statistics\%29}
#'
#' @return
#'   An object of class 'xgb.cv.synchronous' with the following elements:
#'   - `call`: Function call.
#'   - `params`: Parameters that were passed to the xgboost library. Note that it does not
#'     capture parameters changed by the [xgb.cb.reset.parameters()] callback.
#'   - `evaluation_log`: Evaluation history stored as a `data.table` with the
#'     first column corresponding to iteration number and the rest corresponding to the
#'     CV-based evaluation means and standard deviations for the training and test CV-sets.
#'     It is created by the [xgb.cb.evaluation.log()] callback.
#'   - `niter`: Number of boosting iterations.
#'   - `nfeatures`: Number of features in training data.
#'   - `folds`: The list of CV folds' indices - either those passed through the `folds`
#'      parameter or randomly generated.
#'   - `best_iteration`: Iteration number with the best evaluation metric value
#'      (only available with early stopping).
#'
#'   Plus other potential elements that are the result of callbacks, such as a list `cv_predict` with
#'   a sub-element `pred` when passing `prediction = TRUE`, which is added by the [xgb.cb.cv.predict()]
#'   callback (note that one can also pass it manually under `callbacks` with different settings,
#'   such as saving also the models created during cross validation); or a list `early_stop` which
#'   will contain elements such as `best_iteration` when using the early stopping callback ([xgb.cb.early.stop()]).
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#'
#' cv <- xgb.cv(
#'   data = dtrain,
#'   nrounds = 3,
#'   params = xgb.params(
#'     nthread = 2,
#'     max_depth = 3,
#'     objective = "binary:logistic"
#'   ),
#'   nfold = 5,
#'   metrics = list("rmse","auc")
#' )
#' print(cv)
#' print(cv, verbose = TRUE)
#'
#' @export
xgb.cv <- function(params = xgb.params(), data, nrounds, nfold,
                   prediction = FALSE, showsd = TRUE, metrics = list(),
                   objective = NULL, custom_metric = NULL, stratified = "auto",
                   folds = NULL, train_folds = NULL, verbose = TRUE, print_every_n = 1L,
                   early_stopping_rounds = NULL, maximize = NULL, callbacks = list(), ...) {
  check.deprecation(deprecated_train_params, match.call(), ...)

  stopifnot(inherits(data, "xgb.DMatrix"))
  if (inherits(data, "xgb.DMatrix") && .Call(XGCheckNullPtr_R, data)) {
    stop("'data' is an invalid 'xgb.DMatrix' object. Must be constructed again.")
  }

  params <- check.booster.params(params)
  # TODO: should we deprecate the redundant 'metrics' parameter?
  for (m in metrics)
    params <- c(params, list("eval_metric" = m))

  tmp <- check.custom.obj(params, objective)
  params <- tmp$params
  objective <- tmp$objective
  tmp <- check.custom.eval(params, custom_metric, maximize, early_stopping_rounds, callbacks)
  params <- tmp$params
  custom_metric <- tmp$custom_metric

  if (stratified == "auto") {
    if (is.character(params$objective)) {
      stratified <- (
        (params$objective %in% .CLASSIFICATION_OBJECTIVES())
        && !(params$objective %in% .RANKING_OBJECTIVES())
      )
    } else {
      stratified <- FALSE
    }
  }

  # Check the labels and groups
  cv_label <- getinfo(data, "label")
  cv_group <- getinfo(data, "group")
  if (!is.null(train_folds) && NROW(cv_group)) {
    stop("'train_folds' is not supported for DMatrix object with 'group' field.")
  }

  # CV folds
  if (!is.null(folds)) {
    if (!is.list(folds) || length(folds) < 2)
      stop("'folds' must be a list with 2 or more elements that are vectors of indices for each CV-fold")
    nfold <- length(folds)
  } else {
    if (nfold <= 1)
      stop("'nfold' must be > 1")
    folds <- generate.cv.folds(nfold, nrow(data), stratified, cv_label, cv_group, params)
  }

  # Callbacks
  tmp <- .process.callbacks(callbacks, is_cv = TRUE)
  callbacks <- tmp$callbacks
  cb_names <- tmp$cb_names
  rm(tmp)

  # Early stopping callback
  if (!is.null(early_stopping_rounds) && !("early_stop" %in% cb_names)) {
    callbacks <- add.callback(
      callbacks,
      xgb.cb.early.stop(
        early_stopping_rounds,
        maximize = maximize,
        verbose = verbose
      ),
      as_first_elt = TRUE
    )
  }
  # verbosity & evaluation printing callback:
  params <- c(params, list(silent = 1))
  print_every_n <- max(as.integer(print_every_n), 1L)
  if (verbose && !("print_evaluation" %in% cb_names)) {
    callbacks <- add.callback(callbacks, xgb.cb.print.evaluation(print_every_n, showsd = showsd))
  }
  # evaluation log callback: always is on in CV
  if (!("evaluation_log" %in% cb_names)) {
    callbacks <- add.callback(callbacks, xgb.cb.evaluation.log())
  }
  # CV-predictions callback
  if (prediction && !("cv_predict" %in% cb_names)) {
    callbacks <- add.callback(callbacks, xgb.cb.cv.predict(save_models = FALSE))
  }

  # create the booster-folds
  # train_folds
  dall <- data
  bst_folds <- lapply(seq_along(folds), function(k) {
    dtest <- xgb.slice.DMatrix(dall, folds[[k]], allow_groups = TRUE)
    # code originally contributed by @RolandASc on stackoverflow
    if (is.null(train_folds))
       dtrain <- xgb.slice.DMatrix(dall, unlist(folds[-k]), allow_groups = TRUE)
    else
       dtrain <- xgb.slice.DMatrix(dall, train_folds[[k]], allow_groups = TRUE)
    if (!is.null(attributes(folds[[k]])$group_test)) {
      setinfo(dtest, "group", attributes(folds[[k]])$group_test)
      setinfo(dtrain, "group", attributes(folds[[k]])$group_train)
    }
    bst <- xgb.Booster(
      params = params,
      cachelist = list(dtrain, dtest),
      modelfile = NULL
    )
    bst <- bst$bst
    list(dtrain = dtrain, bst = bst, evals = list(train = dtrain, test = dtest), index = folds[[k]])
  })

  # extract parameters that can affect the relationship b/w #trees and #iterations
  num_class <- max(as.numeric(NVL(params[['num_class']], 1)), 1) # nolint

  # those are fixed for CV (no training continuation)
  begin_iteration <- 1
  end_iteration <- nrounds

  .execute.cb.before.training(
    callbacks,
    bst_folds,
    dall,
    NULL,
    begin_iteration,
    end_iteration
  )

  # synchronous CV boosting: run CV folds' models within each iteration
  for (iteration in begin_iteration:end_iteration) {

    .execute.cb.before.iter(
      callbacks,
      bst_folds,
      dall,
      NULL,
      iteration
    )

    msg <- lapply(bst_folds, function(fd) {
      xgb.iter.update(
        bst = fd$bst,
        dtrain = fd$dtrain,
        iter = iteration - 1,
        objective = objective
      )
      xgb.iter.eval(
        bst = fd$bst,
        evals = fd$evals,
        iter = iteration - 1,
        custom_metric = custom_metric
      )
    })
    msg <- simplify2array(msg)

    should_stop <- .execute.cb.after.iter(
      callbacks,
      bst_folds,
      dall,
      NULL,
      iteration,
      msg
    )

    if (should_stop) break
  }
  cb_outputs <- .execute.cb.after.training(
    callbacks,
    bst_folds,
    dall,
    NULL,
    iteration,
    msg
  )

  # the CV result
  ret <- list(
    call = match.call(),
    params = params,
    niter = iteration,
    nfeatures = ncol(dall),
    folds = folds
  )
  ret <- c(ret, cb_outputs)

  class(ret) <- 'xgb.cv.synchronous'
  return(invisible(ret))
}



#' Print xgb.cv result
#'
#' Prints formatted results of [xgb.cv()].
#'
#' @param x An `xgb.cv.synchronous` object.
#' @param verbose Whether to print detailed data.
#' @param ... Passed to `data.table.print()`.
#'
#' @details
#' When not verbose, it would only print the evaluation results,
#' including the best iteration (when available).
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' train <- agaricus.train
#' cv <- xgb.cv(
#'   data = xgb.DMatrix(train$data, label = train$label),
#'   nfold = 5,
#'   nrounds = 2,
#'   params = xgb.params(
#'     max_depth = 2,
#'     nthread = 2,
#'     objective = "binary:logistic"
#'   )
#' )
#' print(cv)
#' print(cv, verbose = TRUE)
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
      cat('  ',
          paste(names(x$params),
                paste0('"', unlist(x$params), '"'),
                sep = ' = ', collapse = ', '), '\n', sep = '')
    }

    for (n in c('niter', 'best_iteration')) {
      if (is.null(x$early_stop[[n]]))
        next
      cat(n, ': ', x$early_stop[[n]], '\n', sep = '')
    }

    if (!is.null(x$cv_predict$pred)) {
      cat('pred:\n')
      str(x$cv_predict$pred)
    }
  }

  if (verbose)
    cat('evaluation_log:\n')
  print(x$evaluation_log, row.names = FALSE, ...)

  if (!is.null(x$early_stop$best_iteration)) {
    cat('Best iteration:\n')
    print(x$evaluation_log[x$early_stop$best_iteration], row.names = FALSE, ...)
  }
  invisible(x)
}
