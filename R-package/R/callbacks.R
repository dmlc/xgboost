.reserved_cb_names <- c("names", "class", "call", "params", "niter", "nfeatures", "folds")

#' XGBoost Callback Constructor
#'
#' Constructor for defining the structure of callback functions that can be executed
#' at different stages of model training (before / after training, before / after each boosting
#' iteration).
#'
#' @details
#' Arguments that will be passed to the supplied functions are as follows:
#' - env The same environment that is passed under argument `env`.
#'
#'   It may be modified by the functions in order to e.g. keep tracking of what happens
#'   across iterations or similar.
#'
#'   This environment is only used by the functions supplied to the callback, and will
#'   not be kept after the model fitting function terminates (see parameter `f_after_training`).
#'
#' - model The booster object when using [xgb.train()], or the folds when using [xgb.cv()].
#'
#'   For [xgb.cv()], folds are a list with a structure as follows:
#'     - `dtrain`: The training data for the fold (as an `xgb.DMatrix` object).
#'     - `bst`: Rhe `xgb.Booster` object for the fold.
#'     - `evals`: A list containing two DMatrices, with names `train` and `test`
#'       (`test` is the held-out data for the fold).
#'     - `index`: The indices of the hold-out data for that fold (base-1 indexing),
#'       from which the `test` entry in `evals` was obtained.
#'
#'   This object should **not** be in-place modified in ways that conflict with the
#'   training (e.g. resetting the parameters for a training update in a way that resets
#'   the number of rounds to zero in order to overwrite rounds).
#'
#'   Note that any R attributes that are assigned to the booster during the callback functions,
#'   will not be kept thereafter as the booster object variable is not re-assigned during
#'   training. It is however possible to set C-level attributes of the booster through
#'   [xgb.attr()] or [xgb.attributes()], which should remain available for the rest
#'   of the iterations and after the training is done.
#'
#'   For keeping variables across iterations, it's recommended to use `env` instead.
#' - data The data to which the model is being fit, as an `xgb.DMatrix` object.
#'
#'   Note that, for [xgb.cv()], this will be the full data, while data for the specific
#'   folds can be found in the `model` object.
#' - evals The evaluation data, as passed under argument `evals` to [xgb.train()].
#'
#'   For [xgb.cv()], this will always be `NULL`.
#' - begin_iteration Index of the first boosting iteration that will be executed (base-1 indexing).
#'
#'   This will typically be '1', but when using training continuation, depending on the
#'   parameters for updates, boosting rounds will be continued from where the previous
#'   model ended, in which case this will be larger than 1.
#'
#' - end_iteration Index of the last boostign iteration that will be executed
#'   (base-1 indexing, inclusive of this end).
#'
#'   It should match with argument `nrounds` passed to [xgb.train()] or [xgb.cv()].
#'
#'   Note that boosting might be interrupted before reaching this last iteration, for
#'   example by using the early stopping callback [xgb.cb.early.stop()].
#' - iteration Index of the iteration number that is being executed (first iteration
#'   will be the same as parameter `begin_iteration`, then next one will add +1, and so on).
#'
#' - iter_feval Evaluation metrics for `evals` that were supplied, either
#'   determined by the objective, or by parameter `custom_metric`.
#'
#'   For [xgb.train()], this will be a named vector with one entry per element in
#'   `evals`, where the names are determined as 'evals name' + '-' + 'metric name' - for
#'   example, if `evals` contains an entry named "tr" and the metric is "rmse",
#'   this will be a one-element vector with name "tr-rmse".
#'
#'   For [xgb.cv()], this will be a 2d matrix with dimensions `[length(evals), nfolds]`,
#'   where the row names will follow the same naming logic as the one-dimensional vector
#'   that is passed in [xgb.train()].
#'
#'   Note that, internally, the built-in callbacks such as [xgb.cb.print.evaluation] summarize
#'   this table by calculating the row-wise means and standard deviations.
#'
#' - final_feval The evaluation results after the last boosting round is executed
#'   (same format as `iter_feval`, and will be the exact same input as passed under
#'   `iter_feval` to the last round that is executed during model fitting).
#'
#' - prev_cb_res Result from a previous run of a callback sharing the same name
#'   (as given by parameter `cb_name`) when conducting training continuation, if there
#'   was any in the booster R attributes.
#'
#'   Sometimes, one might want to append the new results to the previous one, and this will
#'   be done automatically by the built-in callbacks such as [xgb.cb.evaluation.log],
#'   which will append the new rows to the previous table.
#'
#'   If no such previous callback result is available (which it never will when fitting
#'   a model from start instead of updating an existing model), this will be `NULL`.
#'
#'   For [xgb.cv()], which doesn't support training continuation, this will always be `NULL`.
#'
#' The following names (`cb_name` values) are reserved for internal callbacks:
#' - print_evaluation
#' - evaluation_log
#' - reset_parameters
#' - early_stop
#' - save_model
#' - cv_predict
#' - gblinear_history
#'
#' The following names are reserved for other non-callback attributes:
#' - names
#' - class
#' - call
#' - params
#' - niter
#' - nfeatures
#' - folds
#'
#' When using the built-in early stopping callback ([xgb.cb.early.stop]), said callback
#' will always be executed before the others, as it sets some booster C-level attributes
#' that other callbacks might also use. Otherwise, the order of execution will match with
#' the order in which the callbacks are passed to the model fitting function.
#'
#' @param cb_name Name for the callback.
#'
#'   If the callback produces some non-NULL result (from executing the function passed under
#'   `f_after_training`), that result will be added as an R attribute to the resulting booster
#'   (or as a named element in the result of CV), with the attribute name specified here.
#'
#'   Names of callbacks must be unique - i.e. there cannot be two callbacks with the same name.
#' @param env An environment object that will be passed to the different functions in the callback.
#'   Note that this environment will not be shared with other callbacks.
#' @param f_before_training A function that will be executed before the training has started.
#'
#'   If passing `NULL` for this or for the other function inputs, then no function will be executed.
#'
#'   If passing a function, it will be called with parameters supplied as non-named arguments
#'   matching the function signatures that are shown in the default value for each function argument.
#' @param f_before_iter A function that will be executed before each boosting round.
#'
#'   This function can signal whether the training should be finalized or not, by outputting
#'   a value that evaluates to `TRUE` - i.e. if the output from the function provided here at
#'   a given round is `TRUE`, then training will be stopped before the current iteration happens.
#'
#'   Return values of `NULL` will be interpreted as `FALSE`.
#' @param f_after_iter A function that will be executed after each boosting round.
#'
#'   This function can signal whether the training should be finalized or not, by outputting
#'   a value that evaluates to `TRUE` - i.e. if the output from the function provided here at
#'   a given round is `TRUE`, then training will be stopped at that round.
#'
#'   Return values of `NULL` will be interpreted as `FALSE`.
#' @param f_after_training A function that will be executed after training is finished.
#'
#'   This function can optionally output something non-NULL, which will become part of the R
#'   attributes of the booster (assuming one passes `keep_extra_attributes=TRUE` to [xgb.train()])
#'   under the name supplied for parameter `cb_name` imn the case of [xgb.train()]; or a part
#'   of the named elements in the result of [xgb.cv()].
#' @return An `xgb.Callback` object, which can be passed to [xgb.train()] or [xgb.cv()].
#'
#' @seealso Built-in callbacks:
#' - [xgb.cb.print.evaluation]
#' - [xgb.cb.evaluation.log]
#' - [xgb.cb.reset.parameters]
#' - [xgb.cb.early.stop]
#' - [xgb.cb.save.model]
#' - [xgb.cb.cv.predict]
#' - [xgb.cb.gblinear.history]
#
#' @examples
#' # Example constructing a custom callback that calculates
#' # squared error on the training data (no separate test set),
#' # and outputs the per-iteration results.
#' ssq_callback <- xgb.Callback(
#'   cb_name = "ssq",
#'   f_before_training = function(env, model, data, evals,
#'                                begin_iteration, end_iteration) {
#'     # A vector to keep track of a number at each iteration
#'     env$logs <- rep(NA_real_, end_iteration - begin_iteration + 1)
#'   },
#'   f_after_iter = function(env, model, data, evals, iteration, iter_feval) {
#'     # This calculates the sum of squared errors on the training data.
#'     # Note that this can be better done by passing an 'evals' entry,
#'     # but this demonstrates a way in which callbacks can be structured.
#'     pred <- predict(model, data)
#'     err <- pred - getinfo(data, "label")
#'     sq_err <- sum(err^2)
#'     env$logs[iteration] <- sq_err
#'     cat(
#'       sprintf(
#'         "Squared error at iteration %d: %.2f\n",
#'         iteration, sq_err
#'       )
#'     )
#'
#'     # A return value of 'TRUE' here would signal to finalize the training
#'     return(FALSE)
#'   },
#'   f_after_training = function(env, model, data, evals, iteration,
#'                               final_feval, prev_cb_res) {
#'     return(env$logs)
#'   }
#' )
#'
#' data(mtcars)
#'
#' y <- mtcars$mpg
#' x <- as.matrix(mtcars[, -1])
#'
#' dm <- xgb.DMatrix(x, label = y, nthread = 1)
#' model <- xgb.train(
#'   data = dm,
#'   params = xgb.params(objective = "reg:squarederror", nthread = 1),
#'   nrounds = 5,
#'   callbacks = list(ssq_callback)
#' )
#'
#' # Result from 'f_after_iter' will be available as an attribute
#' attributes(model)$ssq
#' @export
xgb.Callback <- function(
  cb_name = "custom_callback",
  env = new.env(),
  f_before_training = function(env, model, data, evals, begin_iteration, end_iteration) NULL,
  f_before_iter = function(env, model, data, evals, iteration) NULL,
  f_after_iter = function(env, model, data, evals, iteration, iter_feval) NULL,
  f_after_training = function(env, model, data, evals, iteration, final_feval, prev_cb_res) NULL
) {
  stopifnot(is.null(f_before_training) || is.function(f_before_training))
  stopifnot(is.null(f_before_iter) || is.function(f_before_iter))
  stopifnot(is.null(f_after_iter) || is.function(f_after_iter))
  stopifnot(is.null(f_after_training) || is.function(f_after_training))
  stopifnot(is.character(cb_name) && length(cb_name) == 1)

  if (cb_name %in% .reserved_cb_names) {
    stop("Cannot use reserved callback name '", cb_name, "'.")
  }

  out <- list(
    cb_name = cb_name,
    env = env,
    f_before_training = f_before_training,
    f_before_iter = f_before_iter,
    f_after_iter = f_after_iter,
    f_after_training = f_after_training
  )
  class(out) <- "xgb.Callback"
  return(out)
}

.execute.cb.before.training <- function(
  callbacks,
  model,
  data,
  evals,
  begin_iteration,
  end_iteration
) {
  for (callback in callbacks) {
    if (!is.null(callback$f_before_training)) {
      callback$f_before_training(
        callback$env,
        model,
        data,
        evals,
        begin_iteration,
        end_iteration
      )
    }
  }
}

.execute.cb.before.iter <- function(
  callbacks,
  model,
  data,
  evals,
  iteration
) {
  if (!length(callbacks)) {
    return(FALSE)
  }
  out <- sapply(callbacks, function(cb) {
    if (is.null(cb$f_before_iter)) {
      return(FALSE)
    }
    should_stop <- cb$f_before_iter(
      cb$env,
      model,
      data,
      evals,
      iteration
    )
    if (!NROW(should_stop)) {
      should_stop <- FALSE
    } else if (NROW(should_stop) > 1) {
      should_stop <- head(as.logical(should_stop), 1)
    }
    return(should_stop)
  })
  return(any(out))
}

.execute.cb.after.iter <- function(
  callbacks,
  model,
  data,
  evals,
  iteration,
  iter_feval
) {
  if (!length(callbacks)) {
    return(FALSE)
  }
  out <- sapply(callbacks, function(cb) {
    if (is.null(cb$f_after_iter)) {
      return(FALSE)
    }
    should_stop <- cb$f_after_iter(
      cb$env,
      model,
      data,
      evals,
      iteration,
      iter_feval
    )
    if (!NROW(should_stop)) {
      should_stop <- FALSE
    } else if (NROW(should_stop) > 1) {
      should_stop <- head(as.logical(should_stop), 1)
    }
    return(should_stop)
  })
  return(any(out))
}

.execute.cb.after.training <- function(
  callbacks,
  model,
  data,
  evals,
  iteration,
  final_feval,
  prev_cb_res
) {
  if (!length(callbacks)) {
    return(NULL)
  }
  old_cb_res <- attributes(model)
  out <- lapply(callbacks, function(cb) {
    if (is.null(cb$f_after_training)) {
      return(NULL)
    } else {
      return(
        cb$f_after_training(
          cb$env,
          model,
          data,
          evals,
          iteration,
          final_feval,
          getElement(old_cb_res, cb$cb_name)
        )
      )
    }
  })
  names(out) <- sapply(callbacks, function(cb) cb$cb_name)
  if (NROW(out)) {
    out <- out[!sapply(out, is.null)]
  }
  return(out)
}

.summarize.feval <- function(iter_feval, showsd) {
  if (NCOL(iter_feval) > 1L && showsd) {
    stdev <- apply(iter_feval, 1, sd)
  } else {
    stdev <- NULL
  }
  if (NCOL(iter_feval) > 1L) {
    iter_feval <- rowMeans(iter_feval)
  }
  return(list(feval = iter_feval, stdev = stdev))
}

.print.evaluation <- function(iter_feval, showsd, iteration) {
  tmp <- .summarize.feval(iter_feval, showsd)
  msg <- .format_eval_string(iteration, tmp$feval, tmp$stdev)
  cat(msg, '\n')
}

# Format the evaluation metric string
.format_eval_string <- function(iter, eval_res, eval_err = NULL) {
  if (length(eval_res) == 0)
    stop('no evaluation results')
  enames <- names(eval_res)
  if (is.null(enames))
    stop('evaluation results must have names')
  iter <- sprintf('[%d]\t', iter)
  if (!is.null(eval_err)) {
    if (length(eval_res) != length(eval_err))
      stop('eval_res & eval_err lengths mismatch')
    # Note: UTF-8 code for plus/minus sign is U+00B1
    res <- paste0(sprintf("%s:%f\U00B1%f", enames, eval_res, eval_err), collapse = '\t')
  } else {
    res <- paste0(sprintf("%s:%f", enames, eval_res), collapse = '\t')
  }
  return(paste0(iter, res))
}

#' Callback for printing the result of evaluation
#'
#' @description
#' The callback function prints the result of evaluation at every `period` iterations.
#' The initial and the last iteration's evaluations are always printed.
#'
#' Does not leave any attribute in the booster (see [xgb.cb.evaluation.log] for that).
#'
#' @param period Results would be printed every number of periods.
#' @param showsd Whether standard deviations should be printed (when available).
#' @return An `xgb.Callback` object, which can be passed to [xgb.train()] or [xgb.cv()].
#' @seealso [xgb.Callback]
#' @export
xgb.cb.print.evaluation <- function(period = 1, showsd = TRUE) {
  if (length(period) != 1 || period != floor(period) || period < 1) {
    stop("'period' must be a positive integer.")
  }

  xgb.Callback(
    cb_name = "print_evaluation",
    env = as.environment(list(period = period, showsd = showsd, is_first_call = TRUE)),
    f_before_training = NULL,
    f_before_iter = NULL,
    f_after_iter = function(env, model, data, evals, iteration, iter_feval) {
      if (is.null(iter_feval)) {
        return(FALSE)
      }
      if (env$is_first_call || (iteration - 1) %% env$period == 0) {
        .print.evaluation(iter_feval, env$showsd, iteration)
        env$last_printed_iter <- iteration
      }
      env$is_first_call <- FALSE
      return(FALSE)
    },
    f_after_training = function(env, model, data, evals, iteration, final_feval, prev_cb_res) {
      if (is.null(final_feval)) {
        return(NULL)
      }
      if (is.null(env$last_printed_iter) || iteration > env$last_printed_iter) {
        .print.evaluation(final_feval, env$showsd, iteration)
      }
    }
  )
}

#' Callback for logging the evaluation history
#'
#' @details This callback creates a table with per-iteration evaluation metrics (see parameters
#' `evals` and `custom_metric` in [xgb.train()]).
#'
#' Note: in the column names of the final data.table, the dash '-' character is replaced with
#' the underscore '_' in order to make the column names more like regular R identifiers.
#'
#' @return An `xgb.Callback` object, which can be passed to [xgb.train()] or [xgb.cv()].
#' @seealso [xgb.cb.print.evaluation]
#' @export
xgb.cb.evaluation.log <- function() {
  xgb.Callback(
    cb_name = "evaluation_log",
    f_before_training = function(env, model, data, evals, begin_iteration, end_iteration) {
      env$evaluation_log <- vector("list", end_iteration - begin_iteration + 1)
      env$next_log <- 1
    },
    f_before_iter = NULL,
    f_after_iter = function(env, model, data, evals, iteration, iter_feval) {
      tmp <- .summarize.feval(iter_feval, TRUE)
      env$evaluation_log[[env$next_log]] <- list(iter = iteration, metrics = tmp$feval, sds = tmp$stdev)
      env$next_log <- env$next_log + 1
      return(FALSE)
    },
    f_after_training = function(env, model, data, evals, iteration, final_feval, prev_cb_res) {
      if (!NROW(env$evaluation_log)) {
        return(prev_cb_res)
      }
      # in case of early stopping
      if (env$next_log <= length(env$evaluation_log)) {
        env$evaluation_log <- head(env$evaluation_log, env$next_log - 1)
      }

      iters <- data.frame(iter = sapply(env$evaluation_log, function(x) x$iter))
      metrics <- do.call(rbind, lapply(env$evaluation_log, function(x) x$metrics))
      mnames <- gsub("-", "_", names(env$evaluation_log[[1]]$metrics), fixed = TRUE)
      colnames(metrics) <- mnames
      has_sds <- !is.null(env$evaluation_log[[1]]$sds)
      if (has_sds) {
        sds <- do.call(rbind, lapply(env$evaluation_log, function(x) x$sds))
        colnames(sds) <- mnames
        metrics <- lapply(
          mnames,
          function(metric) {
            out <- cbind(metrics[, metric], sds[, metric])
            colnames(out) <- paste0(metric, c("_mean", "_std"))
            return(out)
          }
        )
        metrics <- do.call(cbind, metrics)
      }
      evaluation_log <- cbind(iters, metrics)

      if (!is.null(prev_cb_res)) {
        if (!is.data.table(prev_cb_res)) {
          prev_cb_res <- data.table::as.data.table(prev_cb_res)
        }
        prev_take <- prev_cb_res[prev_cb_res$iter < min(evaluation_log$iter)]
        if (nrow(prev_take)) {
          evaluation_log <- rbind(prev_cb_res, evaluation_log)
        }
      }
      evaluation_log <- data.table::as.data.table(evaluation_log)
      return(evaluation_log)
    }
  )
}

#' Callback for resetting booster parameters at each iteration
#'
#' @details
#' Note that when training is resumed from some previous model, and a function is used to
#' reset a parameter value, the `nrounds` argument in this function would be the
#' the number of boosting rounds in the current training.
#'
#' Does not leave any attribute in the booster.
#'
#' @param new_params List of parameters needed to be reset.
#'   Each element's value must be either a vector of values of length `nrounds`
#'   to be set at each iteration,
#'   or a function of two parameters `learning_rates(iteration, nrounds)`
#'   which returns a new parameter value by using the current iteration number
#'   and the total number of boosting rounds.
#' @return An `xgb.Callback` object, which can be passed to [xgb.train()] or [xgb.cv()].
#' @export
xgb.cb.reset.parameters <- function(new_params) {
  stopifnot(is.list(new_params))
  pnames <- gsub(".", "_", names(new_params), fixed = TRUE)
  not_allowed <- pnames %in%
    c('num_class', 'num_output_group', 'size_leaf_vector', 'updater_seq')
  if (any(not_allowed))
    stop('Parameters ', paste(pnames[not_allowed]), " cannot be changed during boosting.")

  xgb.Callback(
    cb_name = "reset_parameters",
    env = as.environment(list(new_params = new_params)),
    f_before_training = function(env, model, data, evals, begin_iteration, end_iteration) {
      env$end_iteration <- end_iteration

      pnames <- gsub(".", "_", names(env$new_params), fixed = TRUE)
      for (n in pnames) {
        p <- env$new_params[[n]]
        if (is.function(p)) {
          if (length(formals(p)) != 2)
            stop("Parameter '", n, "' is a function but not of two arguments")
        } else if (is.numeric(p) || is.character(p)) {
          if (length(p) != env$end_iteration)
            stop("Length of '", n, "' has to be equal to 'nrounds'")
        } else {
          stop("Parameter '", n, "' is not a function or a vector")
        }
      }
    },
    f_before_iter = function(env, model, data, evals, iteration) {
      params <- lapply(env$new_params, function(p) {
        if (is.function(p)) {
          return(p(iteration, env$end_iteration))
        } else {
          return(p[iteration])
        }
      })

      if (inherits(model, "xgb.Booster")) {
        xgb.model.parameters(model) <- params
      } else {
        for (fd in model) {
          xgb.model.parameters(fd$bst) <- params
        }
      }
      return(FALSE)
    },
    f_after_iter = NULL,
    f_after_training = NULL
  )
}

#' Callback to activate early stopping
#'
#' @description
#' This callback function determines the condition for early stopping.
#'
#' The following attributes are assigned to the booster's object:
#' - `best_score` the evaluation score at the best iteration
#' - `best_iteration` at which boosting iteration the best score has occurred
#' (0-based index for interoperability of binary models)
#'
#' The same values are also stored as R attributes as a result of the callback, plus an additional
#' attribute `stopped_by_max_rounds` which indicates whether an early stopping by the `stopping_rounds`
#' condition occurred. Note that the `best_iteration` that is stored under R attributes will follow
#' base-1 indexing, so it will be larger by '1' than the C-level 'best_iteration' that is accessed
#' through [xgb.attr()] or  [xgb.attributes()].
#'
#' At least one dataset is required in `evals` for early stopping to work.
#'
#' @param stopping_rounds The number of rounds with no improvement in
#'   the evaluation metric in order to stop the training.
#' @param maximize Whether to maximize the evaluation metric.
#' @param metric_name The name of an evaluation column to use as a criteria for early
#'   stopping. If not set, the last column would be used.
#'   Let's say the test data in `evals` was labelled as `dtest`,
#'   and one wants to use the AUC in test data for early stopping regardless of where
#'   it is in the `evals`, then one of the following would need to be set:
#'   `metric_name = 'dtest-auc'` or `metric_name = 'dtest_auc'`.
#'   All dash '-' characters in metric names are considered equivalent to '_'.
#' @param verbose Whether to print the early stopping information.
#' @param keep_all_iter Whether to keep all of the boosting rounds that were produced
#'   in the resulting object. If passing `FALSE`, will only keep the boosting rounds
#'   up to the detected best iteration, discarding the ones that come after.
#' @return An `xgb.Callback` object, which can be passed to [xgb.train()] or [xgb.cv()].
#' @export
xgb.cb.early.stop <- function(
  stopping_rounds,
  maximize = FALSE,
  metric_name = NULL,
  verbose = TRUE,
  keep_all_iter = TRUE
) {
  if (!is.null(metric_name)) {
    stopifnot(is.character(metric_name))
    stopifnot(length(metric_name) == 1L)
  }

  xgb.Callback(
    cb_name = "early_stop",
    env = as.environment(
      list(
        checked_evnames = FALSE,
        stopping_rounds = stopping_rounds,
        maximize = maximize,
        metric_name = metric_name,
        verbose = verbose,
        keep_all_iter = keep_all_iter,
        stopped_by_max_rounds = FALSE
      )
    ),
    f_before_training = function(env, model, data, evals, begin_iteration, end_iteration) {
      if (inherits(model, "xgb.Booster") && !length(evals)) {
        stop("For early stopping, 'evals' must have at least one element")
      }
      env$begin_iteration <- begin_iteration
      return(NULL)
    },
    f_before_iter = function(env, model, data, evals, iteration) NULL,
    f_after_iter = function(env, model, data, evals, iteration, iter_feval) {
      sds <- NULL
      if (NCOL(iter_feval) > 1) {
        tmp <- .summarize.feval(iter_feval, TRUE)
        iter_feval <- tmp$feval
        sds <- tmp$stdev
      }

      if (!env$checked_evnames) {

        eval_names <- gsub('-', '_', names(iter_feval), fixed = TRUE)
        if (!is.null(env$metric_name)) {
          env$metric_idx <- which(gsub('-', '_', env$metric_name, fixed = TRUE) == eval_names)
          if (length(env$metric_idx) == 0)
            stop("'metric_name' for early stopping is not one of the following:\n",
                 paste(eval_names, collapse = ' '), '\n')
        }

        if (is.null(env$metric_name)) {
          if (NROW(iter_feval) == 1) {
            env$metric_idx <- 1L
          } else {
            env$metric_idx <- length(eval_names)
            if (env$verbose)
              cat('Multiple eval metrics are present. Will use ',
                  eval_names[env$metric_idx], ' for early stopping.\n', sep = '')
          }
        }

        env$metric_name <- eval_names[env$metric_idx]

        # maximize is usually NULL when not set in xgb.train and built-in metrics
        if (is.null(env$maximize))
          env$maximize <- grepl('(_auc|_aupr|_map|_ndcg|_pre)', env$metric_name)

        if (env$verbose)
          cat("Will train until ", env$metric_name, " hasn't improved in ",
              env$stopping_rounds, " rounds.\n\n", sep = '')

        env$best_iteration <- env$begin_iteration
        if (env$maximize) {
          env$best_score <- -Inf
        } else {
          env$best_score <- Inf
        }

        if (inherits(model, "xgb.Booster")) {
          best_score <- xgb.attr(model, 'best_score')
          if (NROW(best_score)) env$best_score <- as.numeric(best_score)
          best_iteration <- xgb.attr(model, 'best_iteration')
          if (NROW(best_iteration)) env$best_iteration <- as.numeric(best_iteration) + 1
        }

        env$checked_evnames <- TRUE
      }

      score <- iter_feval[env$metric_idx]
      if ((env$maximize && score > env$best_score) ||
          (!env$maximize && score < env$best_score)) {

        env$best_score <- score
        env$best_iteration <- iteration
        # save the property to attributes, so they will occur in checkpoint
        if (inherits(model, "xgb.Booster")) {
          xgb.attributes(model) <- list(
            best_iteration = env$best_iteration - 1, # convert to 0-based index
            best_score = env$best_score
          )
        }
      } else if (iteration - env$best_iteration >= env$stopping_rounds) {
        if (env$verbose) {
          best_msg <- .format_eval_string(iteration, iter_feval, sds)
          cat("Stopping. Best iteration:\n", best_msg, "\n\n", sep = '')
        }
        env$stopped_by_max_rounds <- TRUE
        return(TRUE)
      }
      return(FALSE)
    },
    f_after_training = function(env, model, data, evals, iteration, final_feval, prev_cb_res) {
      if (inherits(model, "xgb.Booster") && !env$keep_all_iter && env$best_iteration < iteration) {
        # Note: it loses the attributes after being sliced,
        # so they have to be re-assigned afterwards.
        prev_attr <- xgb.attributes(model)
        if (NROW(prev_attr)) {
          suppressWarnings({
            prev_attr <- within(prev_attr, rm("best_score", "best_iteration"))
          })
        }
        .Call(XGBoosterSliceAndReplace_R, xgb.get.handle(model), 0L, env$best_iteration, 1L)
        if (NROW(prev_attr)) {
          xgb.attributes(model) <- prev_attr
        }
      }
      attrs_set <- list(best_iteration = env$best_iteration - 1, best_score = env$best_score)
      if (inherits(model, "xgb.Booster")) {
        xgb.attributes(model) <- attrs_set
      } else {
        for (fd in model) {
          xgb.attributes(fd$bst) <- attrs_set # to use in the cv.predict callback
        }
      }
      return(
        list(
          best_iteration = env$best_iteration,
          best_score = env$best_score,
          stopped_by_max_rounds = env$stopped_by_max_rounds
        )
      )
    }
  )
}

.save.model.w.formatted.name <- function(model, save_name, iteration) {
  # Note: this throws a warning if the name doesn't have anything to format through 'sprintf'
  suppressWarnings({
    save_name <- sprintf(save_name, iteration)
  })
  xgb.save(model, save_name)
}

#' Callback for saving a model file
#'
#' @description
#' This callback function allows to save an xgb-model file, either periodically
#' after each `save_period`'s or at the end.
#'
#' Does not leave any attribute in the booster.
#'
#' @param save_period Save the model to disk after every `save_period` iterations;
#'   0 means save the model at the end.
#' @param save_name The name or path for the saved model file.
#'   It can contain a [sprintf()] formatting specifier to include the integer
#'   iteration number in the file name. E.g., with `save_name = 'xgboost_%04d.model'`,
#'   the file saved at iteration 50 would be named "xgboost_0050.model".
#' @return An `xgb.Callback` object, which can be passed to [xgb.train()],
#'   but **not** to [xgb.cv()].
#' @export
xgb.cb.save.model <- function(save_period = 0, save_name = "xgboost.ubj") {
  if (save_period < 0) {
    stop("'save_period' cannot be negative")
  }
  if (!is.character(save_name) || length(save_name) != 1L) {
    stop("'save_name' must be a single character refering to file name.")
  }

  xgb.Callback(
    cb_name = "save_model",
    env = as.environment(list(save_period = save_period, save_name = save_name, last_save = 0)),
    f_before_training = function(env, model, data, evals, begin_iteration, end_iteration) {
      env$begin_iteration <- begin_iteration
    },
    f_before_iter = NULL,
    f_after_iter = function(env, model, data, evals, iteration, iter_feval) {
      if (env$save_period > 0 && (iteration - env$begin_iteration) %% env$save_period == 0) {
        .save.model.w.formatted.name(model, env$save_name, iteration)
        env$last_save <- iteration
      }
      return(FALSE)
    },
    f_after_training = function(env, model, data, evals, iteration, final_feval, prev_cb_res) {
      if (env$save_period == 0 && iteration > env$last_save) {
        .save.model.w.formatted.name(model, env$save_name, iteration)
      }
    }
  )
}

#' Callback for returning cross-validation based predictions
#'
#' This callback function saves predictions for all of the test folds,
#' and also allows to save the folds' models.
#'
#' @details
#' Predictions are saved inside of the `pred` element, which is either a vector or a matrix,
#' depending on the number of prediction outputs per data row. The order of predictions corresponds
#' to the order of rows in the original dataset. Note that when a custom `folds` list is
#' provided in [xgb.cv()], the predictions would only be returned properly when this list is a
#' non-overlapping list of k sets of indices, as in a standard k-fold CV. The predictions would not be
#' meaningful when user-provided folds have overlapping indices as in, e.g., random sampling splits.
#' When some of the indices in the training dataset are not included into user-provided `folds`,
#' their prediction value would be `NA`.
#'
#' @param save_models A flag for whether to save the folds' models.
#' @param outputmargin Whether to save margin predictions (same effect as passing this
#'   parameter to [predict.xgb.Booster]).
#' @return An `xgb.Callback` object, which can be passed to [xgb.cv()],
#'   but **not** to [xgb.train()].
#' @export
xgb.cb.cv.predict <- function(save_models = FALSE, outputmargin = FALSE) {
  xgb.Callback(
    cb_name = "cv_predict",
    env = as.environment(list(save_models = save_models, outputmargin = outputmargin)),
    f_before_training = function(env, model, data, evals, begin_iteration, end_iteration) {
      if (inherits(model, "xgb.Booster")) {
        stop("'cv.predict' callback is only for 'xgb.cv'.")
      }
    },
    f_before_iter = NULL,
    f_after_iter = NULL,
    f_after_training = function(env, model, data, evals, iteration, final_feval, prev_cb_res) {
      pred <- NULL
      for (fd in model) {
        pr <- predict(
          fd$bst,
          fd$evals[[2L]],
          outputmargin = env$outputmargin
        )
        if (is.null(pred)) {
          if (NCOL(pr) > 1L) {
            pred <- matrix(NA_real_, nrow(data), ncol(pr))
          } else {
            pred <- matrix(NA_real_, nrow(data))
          }
        }
        if (is.matrix(pred)) {
          pred[fd$index, ] <- pr
        } else {
          pred[fd$index] <- pr
        }
      }
      out <- list(pred = pred)
      if (env$save_models) {
        out$models <- lapply(model, function(fd) fd$bst)
      }
      return(out)
    }
  )
}

.list2mat <- function(coef_list, sparse) {
  if (sparse) {
    coef_mat <- methods::new("dgRMatrix")
    coef_mat@p <- as.integer(c(0, cumsum(sapply(coef_list, function(x) length(x@x)))))
    coef_mat@j <- as.integer(unlist(lapply(coef_list, slot, "i")) - 1L)
    coef_mat@x <- unlist(lapply(coef_list, slot, "x"))
    coef_mat@Dim <- as.integer(c(length(coef_list), length(coef_list[[1L]])))
    # Note: function 'xgb.gblinear.history' might later on try to slice by columns
    coef_mat <- methods::as(coef_mat, "CsparseMatrix")
    return(coef_mat)
  } else {
    return(unname(do.call(rbind, coef_list)))
  }
}

.extract.coef <- function(model, sparse) {
  coefs <- .internal.coef.xgb.Booster(model, add_names = FALSE)
  if (NCOL(coefs) > 1L) {
    coefs <- as.vector(coefs)
  }
  if (sparse) {
    coefs <- methods::as(coefs, "sparseVector")
  }
  return(coefs)
}

#' Callback for collecting coefficients history of a gblinear booster
#'
#' @details
#' To keep things fast and simple, gblinear booster does not internally store the history of linear
#' model coefficients at each boosting iteration. This callback provides a workaround for storing
#' the coefficients' path, by extracting them after each training iteration.
#'
#' This callback will construct a matrix where rows are boosting iterations and columns are
#' feature coefficients (same order as when calling [coef.xgb.Booster], with the intercept
#' corresponding to the first column).
#'
#' When there is more than one coefficient per feature (e.g. multi-class classification),
#' the result will be reshaped into a vector where coefficients are arranged first by features and
#' then by class (e.g. first 1 through N coefficients will be for the first class, then
#' coefficients N+1 through 2N for the second class, and so on).
#'
#' If the result has only one coefficient per feature in the data, then the resulting matrix
#' will have column names matching with the feature names, otherwise (when there's more than
#' one coefficient per feature) the names will be composed as 'column name' + ':' + 'class index'
#' (so e.g. column 'c1' for class '0' will be named 'c1:0').
#'
#' With [xgb.train()], the output is either a dense or a sparse matrix.
#' With with [xgb.cv()], it is a list (one element per each fold) of such matrices.
#'
#' Function [xgb.gblinear.history] provides an easy way to retrieve the
#' outputs from this callback.
#'
#' @param sparse When set to `FALSE`/`TRUE`, a dense/sparse matrix is used to store the result.
#'   Sparse format is useful when one expects only a subset of coefficients to be non-zero,
#'   when using the "thrifty" feature selector with fairly small number of top features
#'   selected per iteration.
#' @return An `xgb.Callback` object, which can be passed to [xgb.train()] or [xgb.cv()].
#' @seealso [xgb.gblinear.history], [coef.xgb.Booster].
#' @examples
#' #### Binary classification:
#'
#' ## Keep the number of threads to 1 for examples
#' nthread <- 1
#' data.table::setDTthreads(nthread)
#'
#' # In the iris dataset, it is hard to linearly separate Versicolor class from the rest
#' # without considering the 2nd order interactions:
#' x <- model.matrix(Species ~ .^2, iris)[, -1]
#' colnames(x)
#' dtrain <- xgb.DMatrix(
#'   scale(x),
#'   label = 1 * (iris$Species == "versicolor"),
#'   nthread = nthread
#' )
#' param <- xgb.params(
#'   booster = "gblinear",
#'   objective = "reg:logistic",
#'   eval_metric = "auc",
#'   reg_lambda = 0.0003,
#'   reg_alpha = 0.0003,
#'   nthread = nthread
#' )
#'
#' # For 'shotgun', which is a default linear updater, using high learning_rate values may result in
#' # unstable behaviour in some datasets. With this simple dataset, however, the high learning
#' # rate does not break the convergence, but allows us to illustrate the typical pattern of
#' # "stochastic explosion" behaviour of this lock-free algorithm at early boosting iterations.
#' bst <- xgb.train(
#'   c(param, list(learning_rate = 1.)),
#'   dtrain,
#'   evals = list(tr = dtrain),
#'   nrounds = 200,
#'   callbacks = list(xgb.cb.gblinear.history())
#' )
#'
#' # Extract the coefficients' path and plot them vs boosting iteration number:
#' coef_path <- xgb.gblinear.history(bst)
#' matplot(coef_path, type = "l")
#'
#' # With the deterministic coordinate descent updater, it is safer to use higher learning rates.
#' # Will try the classical componentwise boosting which selects a single best feature per round:
#' bst <- xgb.train(
#'   c(
#'     param,
#'     xgb.params(
#'       learning_rate = 0.8,
#'       updater = "coord_descent",
#'       feature_selector = "thrifty",
#'       top_k = 1
#'     )
#'   ),
#'   dtrain,
#'   evals = list(tr = dtrain),
#'   nrounds = 200,
#'   callbacks = list(xgb.cb.gblinear.history())
#' )
#' matplot(xgb.gblinear.history(bst), type = "l")
#' #  Componentwise boosting is known to have similar effect to Lasso regularization.
#' # Try experimenting with various values of top_k, learning_rate, nrounds,
#' # as well as different feature_selectors.
#'
#' # For xgb.cv:
#' bst <- xgb.cv(
#'   c(
#'     param,
#'     xgb.params(
#'       learning_rate = 0.8,
#'       updater = "coord_descent",
#'       feature_selector = "thrifty",
#'       top_k = 1
#'     )
#'   ),
#'   dtrain,
#'   nfold = 5,
#'   nrounds = 100,
#'   callbacks = list(xgb.cb.gblinear.history())
#' )
#' # coefficients in the CV fold #3
#' matplot(xgb.gblinear.history(bst)[[3]], type = "l")
#'
#'
#' #### Multiclass classification:
#' dtrain <- xgb.DMatrix(scale(x), label = as.numeric(iris$Species) - 1, nthread = nthread)
#'
#' param <- xgb.params(
#'   booster = "gblinear",
#'   objective = "multi:softprob",
#'   num_class = 3,
#'   reg_lambda = 0.0003,
#'   reg_alpha = 0.0003,
#'   nthread = nthread
#' )
#'
#' # For the default linear updater 'shotgun' it sometimes is helpful
#' # to use smaller learning_rate to reduce instability
#' bst <- xgb.train(
#'   c(param, list(learning_rate = 0.5)),
#'   dtrain,
#'   evals = list(tr = dtrain),
#'   nrounds = 50,
#'   callbacks = list(xgb.cb.gblinear.history())
#' )
#'
#' # Will plot the coefficient paths separately for each class:
#' matplot(xgb.gblinear.history(bst, class_index = 0), type = "l")
#' matplot(xgb.gblinear.history(bst, class_index = 1), type = "l")
#' matplot(xgb.gblinear.history(bst, class_index = 2), type = "l")
#'
#' # CV:
#' bst <- xgb.cv(
#'   c(param, list(learning_rate = 0.5)),
#'   dtrain,
#'   nfold = 5,
#'   nrounds = 70,
#'   callbacks = list(xgb.cb.gblinear.history(FALSE))
#' )
#' # 1st fold of 1st class
#' matplot(xgb.gblinear.history(bst, class_index = 0)[[1]], type = "l")
#'
#' @export
xgb.cb.gblinear.history <- function(sparse = FALSE) {
  xgb.Callback(
    cb_name = "gblinear_history",
    env = as.environment(list(sparse = sparse)),
    f_before_training = function(env, model, data, evals, begin_iteration, end_iteration) {
      if (!inherits(model, "xgb.Booster")) {
        model <- model[[1L]]$bst
      }
      if (xgb.booster_type(model) != "gblinear") {
        stop("Callback 'xgb.cb.gblinear.history' is only for booster='gblinear'.")
      }
      env$coef_hist <- vector("list", end_iteration - begin_iteration + 1)
      env$next_idx <- 1
    },
    f_before_iter = NULL,
    f_after_iter = function(env, model, data, evals, iteration, iter_feval) {
      if (inherits(model, "xgb.Booster")) {
        coef_this <- .extract.coef(model, env$sparse)
      } else {
        coef_this <- lapply(model, function(fd) .extract.coef(fd$bst, env$sparse))
      }
      env$coef_hist[[env$next_idx]] <- coef_this
      env$next_idx <- env$next_idx + 1
      return(FALSE)
    },
    f_after_training = function(env, model, data, evals, iteration, final_feval, prev_cb_res) {
      # in case of early stopping
      if (env$next_idx <= length(env$coef_hist)) {
        env$coef_hist <- head(env$coef_hist, env$next_idx - 1)
      }

      is_booster <- inherits(model, "xgb.Booster")
      if (is_booster) {
        out <- .list2mat(env$coef_hist, env$sparse)
      } else {
        out <- lapply(
          X = lapply(
            X = seq_along(env$coef_hist[[1]]),
            FUN = function(i) lapply(env$coef_hist, "[[", i)
          ),
          FUN = .list2mat,
          env$sparse
        )
      }
      if (!is.null(prev_cb_res)) {
        if (is_booster) {
          out <- rbind(prev_cb_res, out)
        } else {
          # Note: this case should never be encountered, since training cannot
          # be continued from the result of xgb.cv, but this code should in
          # theory do the job if the situation were to be encountered.
          out <- lapply(
            out,
            function(lst) {
              lapply(
                seq_along(lst),
                function(i) rbind(prev_cb_res[[i]], lst[[i]])
              )
            }
          )
        }
      }
      feature_names <- getinfo(data, "feature_name")
      if (!NROW(feature_names)) {
        feature_names <- paste0("V", seq(1L, ncol(data)))
      }
      expected_ncols <- length(feature_names) + 1
      if (is_booster) {
        mat_ncols <- ncol(out)
      } else {
        mat_ncols <- ncol(out[[1L]])
      }
      if (mat_ncols %% expected_ncols == 0) {
        feature_names <- c("(Intercept)", feature_names)
        n_rep <- mat_ncols / expected_ncols
        if (n_rep > 1) {
          feature_names <- unlist(
            lapply(
              seq(1, n_rep),
              function(cl) paste(feature_names, cl - 1, sep = ":")
            )
          )
        }
        if (is_booster) {
          colnames(out) <- feature_names
        } else {
          out <- lapply(
            out,
            function(mat) {
              colnames(mat) <- feature_names
              return(mat)
            }
          )
        }
      }
      return(out)
    }
  )
}

#' Extract gblinear coefficients history
#'
#' A helper function to extract the matrix of linear coefficients' history
#' from a gblinear model created while using the [xgb.cb.gblinear.history]
#' callback (which must be added manually as by default it is not used).
#'
#' @details
#' Note that this is an R-specific function that relies on R attributes that
#' are not saved when using XGBoost's own serialization functions like [xgb.load()]
#' or [xgb.load.raw()].
#'
#' In order for a serialized model to be accepted by this function, one must use R
#' serializers such as [saveRDS()].
#' @param model Either an `xgb.Booster` or a result of [xgb.cv()], trained
#'   using the [xgb.cb.gblinear.history] callback, but **not** a booster
#'   loaded from [xgb.load()] or [xgb.load.raw()].
#' @param class_index zero-based class index to extract the coefficients for only that
#'   specific class in a multinomial multiclass model. When it is `NULL`, all the
#'   coefficients are returned. Has no effect in non-multiclass models.
#'
#' @return
#' For an [xgb.train()] result, a matrix (either dense or sparse) with the columns
#' corresponding to iteration's coefficients and the rows corresponding to boosting iterations.
#'
#' For an [xgb.cv()] result, a list of such matrices is returned with the elements
#' corresponding to CV folds.
#'
#' When there is more than one coefficient per feature (e.g. multi-class classification)
#' and `class_index` is not provided,
#' the result will be reshaped into a vector where coefficients are arranged first by features and
#' then by class (e.g. first 1 through N coefficients will be for the first class, then
#' coefficients N+1 through 2N for the second class, and so on).
#' @seealso [xgb.cb.gblinear.history], [coef.xgb.Booster].
#' @export
xgb.gblinear.history <- function(model, class_index = NULL) {

  if (!(inherits(model, "xgb.Booster") ||
        inherits(model, "xgb.cv.synchronous")))
    stop("model must be an object of either xgb.Booster or xgb.cv.synchronous class")
  is_cv <- inherits(model, "xgb.cv.synchronous")

  if (!is_cv) {
    coef_path <- getElement(attributes(model), "gblinear_history")
  } else {
    coef_path <- getElement(model, "gblinear_history")
  }
  if (is.null(coef_path)) {
    stop("model must be trained while using the xgb.cb.gblinear.history() callback")
  }

  if (!is_cv) {
    num_class <- xgb.num_class(model)
    num_feat <- xgb.num_feature(model)
  } else {
    # in case of CV, the object is expected to have this info
    if (model$params$booster != "gblinear")
      stop("It does not appear to be a gblinear model")
    num_class <- NVL(model$params$num_class, 1)
    num_feat <- model$nfeatures
    if (is.null(num_feat))
      stop("This xgb.cv result does not have nfeatures info")
  }

  if (!is.null(class_index) &&
      num_class > 1 &&
      (class_index[1] < 0 || class_index[1] >= num_class))
    stop("class_index has to be within [0,", num_class - 1, "]")

  if (!is.null(class_index) && num_class > 1) {
    seq_take <- seq(1 + class_index * (num_feat + 1), (class_index + 1) * (num_feat + 1))
    coef_path <- if (is.list(coef_path)) {
      lapply(coef_path, function(x) x[, seq_take])
    } else {
      coef_path <- coef_path[, seq_take]
    }
  }
  return(coef_path)
}

.callbacks.only.train <- "save_model"
.callbacks.only.cv <- "cv_predict"

.process.callbacks <- function(callbacks, is_cv) {
  if (inherits(callbacks, "xgb.Callback")) {
    callbacks <- list(callbacks)
  }
  if (!is.list(callbacks)) {
    stop("'callbacks' must be a list.")
  }
  cb_names <- character()
  if (length(callbacks)) {
    is_callback <- sapply(callbacks, inherits, "xgb.Callback")
    if (!all(is_callback)) {
      stop("Entries in 'callbacks' must be 'xgb.Callback' objects.")
    }
    cb_names <- sapply(callbacks, function(cb) cb$cb_name)
    if (length(cb_names) != length(callbacks)) {
      stop("Passed invalid callback(s).")
    }
    if (anyDuplicated(cb_names) > 0) {
      stop("Callbacks must have unique names.")
    }
    if (is_cv) {
      if (any(.callbacks.only.train %in% cb_names)) {
        stop(
          "Passed callback(s) not supported for 'xgb.cv': ",
          paste(intersect(.callbacks.only.train, cb_names), collapse = ", ")
        )
      }
    } else {
      if (any(.callbacks.only.cv %in% cb_names)) {
        stop(
          "Passed callback(s) not supported for 'xgb.train': ",
          paste(intersect(.callbacks.only.cv, cb_names), collapse = ", ")
        )
      }
    }
    # Early stopping callback needs to be executed before the others
    if ("early_stop" %in% cb_names) {
      mask <- cb_names == "early_stop"
      callbacks <- c(list(callbacks[[which(mask)]]), callbacks[!mask])
    }
  }
  return(list(callbacks = callbacks, cb_names = cb_names))
}

# Note: don't try to use functions like 'append', as they will
# merge the elements of the different callbacks into a single list.
add.callback <- function(callbacks, cb, as_first_elt = FALSE) {
  if (!as_first_elt) {
    callbacks[[length(callbacks) + 1]] <- cb
    return(callbacks)
  } else {
    if (!length(callbacks)) {
      return(list(cb))
    }
    new_cb <- vector("list", length(callbacks) + 1)
    new_cb[[1]] <- cb
    new_cb[seq(2, length(new_cb))] <- callbacks
    return(new_cb)
  }
}

has.callbacks <- function(callbacks, cb_name) {
  cb_names <- sapply(callbacks, function(cb) cb$name)
  return(cb_name %in% cb_names)
}
