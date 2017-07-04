#' Callback closures for booster training.
#'
#' These are used to perform various service tasks either during boosting iterations or at the end.
#' This approach helps to modularize many of such tasks without bloating the main training methods, 
#' and it offers .
#' 
#' @details
#' By default, a callback function is run after each boosting iteration.
#' An R-attribute \code{is_pre_iteration} could be set for a callback to define a pre-iteration function.
#' 
#' When a callback function has \code{finalize} parameter, its finalizer part will also be run after 
#' the boosting is completed.
#' 
#' WARNING: side-effects!!! Be aware that these callback functions access and modify things in 
#' the environment from which they are called from, which is a fairly uncommon thing to do in R.
#' 
#' To write a custom callback closure, make sure you first understand the main concepts about R envoronments.
#' Check either R documentation on \code{\link[base]{environment}} or the 
#' \href{http://adv-r.had.co.nz/Environments.html}{Environments chapter} from the "Advanced R" 
#' book by Hadley Wickham. Further, the best option is to read the code of some of the existing callbacks -
#' choose ones that do something similar to what you want to achieve. Also, you would need to get familiar 
#' with the objects available inside of the \code{xgb.train} and \code{xgb.cv} internal environments.
#' 
#' @seealso
#' \code{\link{cb.print.evaluation}},
#' \code{\link{cb.evaluation.log}},
#' \code{\link{cb.reset.parameters}},
#' \code{\link{cb.early.stop}},
#' \code{\link{cb.save.model}},
#' \code{\link{cb.cv.predict}},
#' \code{\link{xgb.train}},
#' \code{\link{xgb.cv}}
#' 
#' @name callbacks
NULL

#
# Callbacks -------------------------------------------------------------------
# 

#' Callback closure for printing the result of evaluation
#' 
#' @param period  results would be printed every number of periods
#' @param showsd  whether standard deviations should be printed (when available)
#' 
#' @details
#' The callback function prints the result of evaluation at every \code{period} iterations.
#' The initial and the last iteration's evaluations are always printed.
#' 
#' Callback function expects the following values to be set in its calling frame:
#' \code{bst_evaluation} (also \code{bst_evaluation_err} when available),
#' \code{iteration},
#' \code{begin_iteration},
#' \code{end_iteration}.
#' 
#' @seealso
#' \code{\link{callbacks}}
#' 
#' @export
cb.print.evaluation <- function(period = 1, showsd = TRUE) {
  
  callback <- function(env = parent.frame()) {
    if (length(env$bst_evaluation) == 0 ||
        period == 0 ||
        NVL(env$rank, 0) != 0 )
      return()
    
    i <- env$iteration 
    if ((i-1) %% period == 0 ||
        i == env$begin_iteration ||
        i == env$end_iteration) {
      stdev <- if (showsd) env$bst_evaluation_err else NULL
      msg <- format.eval.string(i, env$bst_evaluation, stdev)
      cat(msg, '\n')
    }
  }
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.print.evaluation'
  callback
}


#' Callback closure for logging the evaluation history
#' 
#' @details
#' This callback function appends the current iteration evaluation results \code{bst_evaluation}
#' available in the calling parent frame to the \code{evaluation_log} list in a calling frame.
#' 
#' The finalizer callback (called with \code{finalize = TURE} in the end) converts 
#' the \code{evaluation_log} list into a final data.table.
#' 
#' The iteration evaluation result \code{bst_evaluation} must be a named numeric vector. 
#' 
#' Note: in the column names of the final data.table, the dash '-' character is replaced with 
#' the underscore '_' in order to make the column names more like regular R identifiers.
#' 
#' Callback function expects the following values to be set in its calling frame:
#' \code{evaluation_log},
#' \code{bst_evaluation},
#' \code{iteration}.
#' 
#' @seealso
#' \code{\link{callbacks}}
#' 
#' @export
cb.evaluation.log <- function() {

  mnames <- NULL
  
  init <- function(env) {
    if (!is.list(env$evaluation_log))
      stop("'evaluation_log' has to be a list")
    mnames <<- names(env$bst_evaluation)
    if (is.null(mnames) || any(mnames == ""))
      stop("bst_evaluation must have non-empty names")
    
    mnames <<- gsub('-', '_', names(env$bst_evaluation))
    if(!is.null(env$bst_evaluation_err))
      mnames <<- c(paste0(mnames, '_mean'), paste0(mnames, '_std'))
  }
  
  finalizer <- function(env) {
    env$evaluation_log <- as.data.table(t(simplify2array(env$evaluation_log)))
    setnames(env$evaluation_log, c('iter', mnames))
    
    if(!is.null(env$bst_evaluation_err)) {
      # rearrange col order from _mean,_mean,...,_std,_std,...
      # to be _mean,_std,_mean,_std,...
      len <- length(mnames)
      means <- mnames[seq_len(len/2)]
      stds <- mnames[(len/2 + 1):len]
      cnames <- numeric(len)
      cnames[c(TRUE, FALSE)] <- means
      cnames[c(FALSE, TRUE)] <- stds
      env$evaluation_log <- env$evaluation_log[, c('iter', cnames), with = FALSE]
    }
  }
  
  callback <- function(env = parent.frame(), finalize = FALSE) {
    if (is.null(mnames))
      init(env)

    if (finalize)
      return(finalizer(env))
    
    ev <- env$bst_evaluation
    if(!is.null(env$bst_evaluation_err))
      ev <- c(ev, env$bst_evaluation_err)
    env$evaluation_log <- c(env$evaluation_log, 
                            list(c(iter = env$iteration, ev)))
  }
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.evaluation.log'
  callback
}

#' Callback closure for restetting the booster's parameters at each iteration.
#' 
#' @param new_params a list where each element corresponds to a parameter that needs to be reset.
#'        Each element's value must be either a vector of values of length \code{nrounds} 
#'        to be set at each iteration, 
#'        or a function of two parameters \code{learning_rates(iteration, nrounds)} 
#'        which returns a new parameter value by using the current iteration number 
#'        and the total number of boosting rounds.
#' 
#' @details 
#' This is a "pre-iteration" callback function used to reset booster's parameters
#' at the beginning of each iteration.
#' 
#' Note that when training is resumed from some previous model, and a function is used to 
#' reset a parameter value, the \code{nround} argument in this function would be the 
#' the number of boosting rounds in the current training.
#'
#' Callback function expects the following values to be set in its calling frame:
#' \code{bst} or \code{bst_folds},
#' \code{iteration},
#' \code{begin_iteration},
#' \code{end_iteration}.
#' 
#' @seealso
#' \code{\link{callbacks}}
#' 
#' @export
cb.reset.parameters <- function(new_params) {

  if (typeof(new_params) != "list") 
    stop("'new_params' must be a list")
  pnames <- gsub("\\.", "_", names(new_params))
  nrounds <- NULL
  
  # run some checks in the begining
  init <- function(env) {
    nrounds <<- env$end_iteration - env$begin_iteration + 1
    
    if (is.null(env$bst) && is.null(env$bst_folds))
      stop("Parent frame has neither 'bst' nor 'bst_folds'")
    
    # Some parameters are not allowed to be changed,
    # since changing them would simply wreck some chaos
    not_allowed <- pnames %in% 
      c('num_class', 'num_output_group', 'size_leaf_vector', 'updater_seq')
    if (any(not_allowed))
      stop('Parameters ', paste(pnames[not_allowed]), " cannot be changed during boosting.")
    
    for (n in pnames) {
      p <- new_params[[n]]
      if (is.function(p)) {
        if (length(formals(p)) != 2)
          stop("Parameter '", n, "' is a function but not of two arguments")
      } else if (is.numeric(p) || is.character(p)) {
        if (length(p) != nrounds)
          stop("Length of '", n, "' has to be equal to 'nrounds'")
      } else {
        stop("Parameter '", n, "' is not a function or a vector")
      }
    }
  }
  
  callback <- function(env = parent.frame()) {
    if (is.null(nrounds))
      init(env)
    
    i <- env$iteration
    pars <- lapply(new_params, function(p) {
      if (is.function(p))
        return(p(i, nrounds))
      p[i]
    })
    
    if (!is.null(env$bst)) {
      xgb.parameters(env$bst$handle) <- pars
    } else {
      for (fd in env$bst_folds)
        xgb.parameters(fd$bst) <- pars
    }
  }
  attr(callback, 'is_pre_iteration') <- TRUE
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.reset.parameters'
  callback
}


#' Callback closure to activate the early stopping.
#' 
#' @param stopping_rounds The number of rounds with no improvement in 
#'        the evaluation metric in order to stop the training.
#' @param maximize whether to maximize the evaluation metric
#' @param metric_name the name of an evaluation column to use as a criteria for early
#'        stopping. If not set, the last column would be used.
#'        Let's say the test data in \code{watchlist} was labelled as \code{dtest}, 
#'        and one wants to use the AUC in test data for early stopping regardless of where 
#'        it is in the \code{watchlist}, then one of the following would need to be set:
#'        \code{metric_name='dtest-auc'} or \code{metric_name='dtest_auc'}.
#'        All dash '-' characters in metric names are considered equivalent to '_'.
#' @param verbose whether to print the early stopping information.
#' 
#' @details
#' This callback function determines the condition for early stopping 
#' by setting the \code{stop_condition = TRUE} flag in its calling frame.
#' 
#' The following additional fields are assigned to the model's R object:
#' \itemize{
#' \item \code{best_score} the evaluation score at the best iteration
#' \item \code{best_iteration} at which boosting iteration the best score has occurred (1-based index)
#' \item \code{best_ntreelimit} to use with the \code{ntreelimit} parameter in \code{predict}.
#'      It differs from \code{best_iteration} in multiclass or random forest settings.
#' }
#' 
#' The Same values are also stored as xgb-attributes:
#' \itemize{
#' \item \code{best_iteration} is stored as a 0-based iteration index (for interoperability of binary models)
#' \item \code{best_msg} message string is also stored.
#' }
#' 
#' At least one data element is required in the evaluation watchlist for early stopping to work.
#'
#' Callback function expects the following values to be set in its calling frame:
#' \code{stop_condition},
#' \code{bst_evaluation},
#' \code{rank},
#' \code{bst} (or \code{bst_folds} and \code{basket}),
#' \code{iteration},
#' \code{begin_iteration},
#' \code{end_iteration},
#' \code{num_parallel_tree}.
#' 
#' @seealso
#' \code{\link{callbacks}},
#' \code{\link{xgb.attr}}
#' 
#' @export
cb.early.stop <- function(stopping_rounds, maximize = FALSE, 
                          metric_name = NULL, verbose = TRUE) {
  # state variables
  best_iteration <- -1
  best_ntreelimit <- -1
  best_score <- Inf
  best_msg <- NULL
  metric_idx <- 1
  
  init <- function(env) {
    if (length(env$bst_evaluation) == 0)
      stop("For early stopping, watchlist must have at least one element")
    
    eval_names <- gsub('-', '_', names(env$bst_evaluation))
    if (!is.null(metric_name)) {
      metric_idx <<- which(gsub('-', '_', metric_name) == eval_names)
      if (length(metric_idx) == 0)
        stop("'metric_name' for early stopping is not one of the following:\n",
             paste(eval_names, collapse = ' '), '\n')
    }
    if (is.null(metric_name) &&
        length(env$bst_evaluation) > 1) {
      metric_idx <<- length(eval_names)
      if (verbose)
        cat('Multiple eval metrics are present. Will use ', 
            eval_names[metric_idx], ' for early stopping.\n', sep = '')
    }
    
    metric_name <<- eval_names[metric_idx]
    
    # maximize is usually NULL when not set in xgb.train and built-in metrics
    if (is.null(maximize))
      maximize <<- grepl('(_auc|_map|_ndcg)', metric_name)

    if (verbose && NVL(env$rank, 0) == 0)
      cat("Will train until ", metric_name, " hasn't improved in ", 
          stopping_rounds, " rounds.\n\n", sep = '')

    best_iteration <<- 1
    if (maximize) best_score <<- -Inf
    
    env$stop_condition <- FALSE
    
    if (!is.null(env$bst)) {
      if (!inherits(env$bst, 'xgb.Booster'))
        stop("'bst' in the parent frame must be an 'xgb.Booster'")
      if (!is.null(best_score <- xgb.attr(env$bst$handle, 'best_score'))) {
        best_score <<- as.numeric(best_score)
        best_iteration <<- as.numeric(xgb.attr(env$bst$handle, 'best_iteration')) + 1
        best_msg <<- as.numeric(xgb.attr(env$bst$handle, 'best_msg'))
      } else {
        xgb.attributes(env$bst$handle) <- list(best_iteration = best_iteration - 1,
                                               best_score = best_score)
      }
    } else if (is.null(env$bst_folds) || is.null(env$basket)) {
      stop("Parent frame has neither 'bst' nor ('bst_folds' and 'basket')")
    }
  }
  
  finalizer <- function(env) {
    if (!is.null(env$bst)) {
      attr_best_score = as.numeric(xgb.attr(env$bst$handle, 'best_score'))
      if (best_score != attr_best_score)
        stop("Inconsistent 'best_score' values between the closure state: ", best_score,
             " and the xgb.attr: ", attr_best_score)
      env$bst$best_iteration = best_iteration
      env$bst$best_ntreelimit = best_ntreelimit
      env$bst$best_score = best_score
    } else {
      env$basket$best_iteration <- best_iteration
      env$basket$best_ntreelimit <- best_ntreelimit
    }
  }

  callback <- function(env = parent.frame(), finalize = FALSE) {
    if (best_iteration < 0)
      init(env)
    
    if (finalize)
      return(finalizer(env))
    
    i <- env$iteration
    score = env$bst_evaluation[metric_idx]
    
    if (( maximize && score > best_score) ||
        (!maximize && score < best_score)) {
      
      best_msg <<- format.eval.string(i, env$bst_evaluation, env$bst_evaluation_err)
      best_score <<- score
      best_iteration <<- i
      best_ntreelimit <<- best_iteration * env$num_parallel_tree
      # save the property to attributes, so they will occur in checkpoint
      if (!is.null(env$bst)) {
        xgb.attributes(env$bst) <- list(
          best_iteration = best_iteration - 1, # convert to 0-based index
          best_score = best_score,
          best_msg = best_msg,
          best_ntreelimit = best_ntreelimit)
      }
    } else if (i - best_iteration >= stopping_rounds) {
      env$stop_condition <- TRUE
      env$end_iteration <- i
      if (verbose && NVL(env$rank, 0) == 0)
        cat("Stopping. Best iteration:\n", best_msg, "\n\n", sep = '')
    }
  }
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.early.stop'
  callback
}


#' Callback closure for saving a model file.
#' 
#' @param save_period save the model to disk after every 
#'        \code{save_period} iterations; 0 means save the model at the end.
#' @param save_name the name or path for the saved model file.
#'        It can contain a \code{\link[base]{sprintf}} formatting specifier 
#'        to include the integer iteration number in the file name.
#'        E.g., with \code{save_name} = 'xgboost_%04d.model', 
#'        the file saved at iteration 50 would be named "xgboost_0050.model".
#' 
#' @details 
#' This callback function allows to save an xgb-model file, either periodically after each \code{save_period}'s or at the end.
#' 
#' Callback function expects the following values to be set in its calling frame:
#' \code{bst},
#' \code{iteration},
#' \code{begin_iteration},
#' \code{end_iteration}.
#' 
#' @seealso
#' \code{\link{callbacks}}
#' 
#' @export
cb.save.model <- function(save_period = 0, save_name = "xgboost.model") {
  
  if (save_period < 0)
    stop("'save_period' cannot be negative")

  callback <- function(env = parent.frame()) {
    if (is.null(env$bst))
      stop("'save_model' callback requires the 'bst' booster object in its calling frame")
    
    if ((save_period > 0 && (env$iteration - env$begin_iteration) %% save_period == 0) ||
        (save_period == 0 && env$iteration == env$end_iteration))
      xgb.save(env$bst, sprintf(save_name, env$iteration))
  }
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.save.model'
  callback
}


#' Callback closure for returning cross-validation based predictions.
#' 
#' @param save_models a flag for whether to save the folds' models.
#' 
#' @details 
#' This callback function saves predictions for all of the test folds,
#' and also allows to save the folds' models.
#' 
#' It is a "finalizer" callback and it uses early stopping information whenever it is available,
#' thus it must be run after the early stopping callback if the early stopping is used.
#' 
#' Callback function expects the following values to be set in its calling frame:
#' \code{bst_folds},
#' \code{basket},
#' \code{data},
#' \code{end_iteration},
#' \code{params},
#' \code{num_parallel_tree},
#' \code{num_class}.
#' 
#' @return 
#' Predictions are returned inside of the \code{pred} element, which is either a vector or a matrix,
#' depending on the number of prediction outputs per data row. The order of predictions corresponds 
#' to the order of rows in the original dataset. Note that when a custom \code{folds} list is 
#' provided in \code{xgb.cv}, the predictions would only be returned properly when this list is a 
#' non-overlapping list of k sets of indices, as in a standard k-fold CV. The predictions would not be 
#' meaningful when user-profided folds have overlapping indices as in, e.g., random sampling splits.
#' When some of the indices in the training dataset are not included into user-provided \code{folds},
#' their prediction value would be \code{NA}.
#' 
#' @seealso
#' \code{\link{callbacks}}
#' 
#' @export
cb.cv.predict <- function(save_models = FALSE) {
  
  finalizer <- function(env) {
    if (is.null(env$basket) || is.null(env$bst_folds))
      stop("'cb.cv.predict' callback requires 'basket' and 'bst_folds' lists in its calling frame")
    
    N <- nrow(env$data)
    pred <- 
      if (env$num_class > 1) {
        matrix(NA_real_, N, env$num_class)
      } else {
        rep(NA_real_, N)
      }

    ntreelimit <- NVL(env$basket$best_ntreelimit, 
                      env$end_iteration * env$num_parallel_tree)
    if (NVL(env$params[['booster']], '') == 'gblinear') {
      ntreelimit <- 0 # must be 0 for gblinear
    }
    for (fd in env$bst_folds) {
      pr <- predict(fd$bst, fd$watchlist[[2]], ntreelimit = ntreelimit, reshape = TRUE)
      if (is.matrix(pred)) {
        pred[fd$index,] <- pr
      } else {
        pred[fd$index] <- pr
      }
    }
    env$basket$pred <- pred
    if (save_models) {
      env$basket$models <- lapply(env$bst_folds, function(fd) {
        xgb.attr(fd$bst, 'niter') <- env$end_iteration - 1
        xgb.Booster.complete(xgb.handleToBooster(fd$bst), saveraw = TRUE)
      })
    }
  }

  callback <- function(env = parent.frame(), finalize = FALSE) {
    if (finalize)
      return(finalizer(env))
  }
  attr(callback, 'call') <- match.call()
  attr(callback, 'name') <- 'cb.cv.predict'
  callback
}


#
# Internal utility functions for callbacks ------------------------------------
# 

# Format the evaluation metric string
format.eval.string <- function(iter, eval_res, eval_err = NULL) {
  if (length(eval_res) == 0)
    stop('no evaluation results')
  enames <- names(eval_res)
  if (is.null(enames))
    stop('evaluation results must have names')
  iter <- sprintf('[%d]\t', iter)
  if (!is.null(eval_err)) {
    if (length(eval_res) != length(eval_err))
      stop('eval_res & eval_err lengths mismatch')
    res <- paste0(sprintf("%s:%f+%f", enames, eval_res, eval_err), collapse = '\t')
  } else {
    res <- paste0(sprintf("%s:%f", enames, eval_res), collapse = '\t')
  }
  return(paste0(iter, res))
}

# Extract callback names from the list of callbacks
callback.names <- function(cb_list) {
  unlist(lapply(cb_list, function(x) attr(x, 'name')))
}

# Extract callback calls from the list of callbacks
callback.calls <- function(cb_list) {
  unlist(lapply(cb_list, function(x) attr(x, 'call')))
}

# Add a callback cb to the list and make sure that 
# cb.early.stop and cb.cv.predict are at the end of the list
# with cb.cv.predict being the last (when present)
add.cb <- function(cb_list, cb) {
  cb_list <- c(cb_list, cb)
  names(cb_list) <- callback.names(cb_list)
  if ('cb.early.stop' %in% names(cb_list)) {
    cb_list <- c(cb_list, cb_list['cb.early.stop'])
    # this removes only the first one
    cb_list['cb.early.stop'] <- NULL 
  }
  if ('cb.cv.predict' %in% names(cb_list)) {
    cb_list <- c(cb_list, cb_list['cb.cv.predict'])
    cb_list['cb.cv.predict'] <- NULL 
  }
  cb_list
}

# Sort callbacks list into categories
categorize.callbacks <- function(cb_list) {
  list(
    pre_iter = Filter(function(x) {
        pre <- attr(x, 'is_pre_iteration')
        !is.null(pre) && pre 
      }, cb_list),
    post_iter = Filter(function(x) {
        pre <- attr(x, 'is_pre_iteration')
        is.null(pre) || !pre
      }, cb_list),
    finalize = Filter(function(x) {
        'finalize' %in% names(formals(x))
      }, cb_list)
  )
}

# Check whether all callback functions with names given by 'query_names' are present in the 'cb_list'.
has.callbacks <- function(cb_list, query_names) {
  if (length(cb_list) < length(query_names))
    return(FALSE)
  if (!is.list(cb_list) ||
      any(sapply(cb_list, class) != 'function')) {
    stop('`cb_list` must be a list of callback functions')
  }
  cb_names <- callback.names(cb_list)
  if (!is.character(cb_names) ||
      length(cb_names) != length(cb_list) ||
      any(cb_names == "")) {
    stop('All callbacks in the `cb_list` must have a non-empty `name` attribute')
  }
  if (!is.character(query_names) ||
      length(query_names) == 0 ||
      any(query_names == "")) {
    stop('query_names must be a non-empty vector of non-empty character names')
  }
  return(all(query_names %in% cb_names))
}
