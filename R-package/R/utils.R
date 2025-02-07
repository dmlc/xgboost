#
# This file is for the low level reusable utility functions
# that are not supposed to be visible to a user.
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
  stop("typeof(x) == ", typeof(x), " is not supported by NVL")
}

# List of classification and ranking objectives
.CLASSIFICATION_OBJECTIVES <- function() {
  return(c('binary:logistic', 'binary:logitraw', 'binary:hinge', 'multi:softmax',
           'multi:softprob', 'rank:pairwise', 'rank:ndcg', 'rank:map'))
}

.RANKING_OBJECTIVES <- function() {
  return(c('rank:pairwise', 'rank:ndcg', 'rank:map'))
}

.OBJECTIVES_NON_DEFAULT_MODE <- function() {
  return(c("reg:logistic", "binary:logitraw", "multi:softmax"))
}

.BINARY_CLASSIF_OBJECTIVES <- function() {
  return(c("binary:logistic", "binary:hinge"))
}

.MULTICLASS_CLASSIF_OBJECTIVES <- function() {
  return("multi:softprob")
}

.SURVIVAL_RIGHT_CENSORING_OBJECTIVES <- function() { # nolint
  return(c("survival:cox", "survival:aft"))
}

.SURVIVAL_ALL_CENSORING_OBJECTIVES <- function() { # nolint
  return("survival:aft")
}

.REGRESSION_OBJECTIVES <- function() {
  return(c(
    "reg:squarederror", "reg:squaredlogerror", "reg:logistic", "reg:pseudohubererror",
    "reg:absoluteerror", "reg:quantileerror", "count:poisson", "reg:gamma", "reg:tweedie"
  ))
}

.MULTI_TARGET_OBJECTIVES <- function() {
  return(c(
    "reg:squarederror", "reg:squaredlogerror", "reg:logistic", "reg:pseudohubererror",
    "reg:quantileerror", "reg:gamma"
  ))
}


#
# Low-level functions for boosting --------------------------------------------
#

# Merges booster params with whatever is provided in ...
# plus runs some checks
check.booster.params <- function(params) {
  if (!identical(class(params), "list"))
    stop("params must be a list")

  # in R interface, allow for '.' instead of '_' in parameter names
  names(params) <- gsub(".", "_", names(params), fixed = TRUE)

  # providing a parameter multiple times makes sense only for 'eval_metric'
  name_freqs <- table(names(params))
  multi_names <- setdiff(names(name_freqs[name_freqs > 1]), 'eval_metric')
  if (length(multi_names) > 0) {
    warning("The following parameters were provided multiple times:\n\t",
            paste(multi_names, collapse = ', '), "\n  Only the last value for each of them will be used.\n")
    # While xgboost internals would choose the last value for a multiple-times parameter,
    # enforce it here in R as well (b/c multi-parameters might be used further in R code,
    # and R takes the 1st value when multiple elements with the same name are present in a list).
    for (n in multi_names) {
      del_idx <- which(n == names(params))
      del_idx <- del_idx[-length(del_idx)]
      params[[del_idx]] <- NULL
    }
  }

  # for multiclass, expect num_class to be set
  if (typeof(params[['objective']]) == "character" &&
      startsWith(NVL(params[['objective']], 'x'), 'multi:') &&
      as.numeric(NVL(params[['num_class']], 0)) < 2) {
        stop("'num_class' > 1 parameter must be set for multiclass classification")
  }

  # monotone_constraints parser
  if (!is.null(params[['monotone_constraints']]) &&
      typeof(params[['monotone_constraints']]) != "character") {
        vec2str <- paste(params[['monotone_constraints']], collapse = ',')
        vec2str <- paste0('(', vec2str, ')')
        params[['monotone_constraints']] <- vec2str
  }

  # interaction constraints parser (convert from list of column indices to string)
  if (!is.null(params[['interaction_constraints']]) &&
      typeof(params[['interaction_constraints']]) != "character") {
    # check input class
    if (!identical(class(params[['interaction_constraints']]), 'list')) stop('interaction_constraints should be class list')
    if (!all(unique(sapply(params[['interaction_constraints']], class)) %in% c('numeric', 'integer'))) {
      stop('interaction_constraints should be a list of numeric/integer vectors')
    }

    # recast parameter as string
    interaction_constraints <- sapply(params[['interaction_constraints']], function(x) paste0('[', paste(x, collapse = ','), ']'))
    params[['interaction_constraints']] <- paste0('[', paste(interaction_constraints, collapse = ','), ']')
  }

  # for evaluation metrics, should generate multiple entries per metric
  if (NROW(params[['eval_metric']]) > 1) {
    eval_metrics <- as.list(params[["eval_metric"]])
    names(eval_metrics) <- rep("eval_metric", length(eval_metrics))
    params_without_ev_metrics <- within(params, rm("eval_metric"))
    params <- c(params_without_ev_metrics, eval_metrics)
  }
  return(params)
}


# Performs some checks related to custom objective function.
check.custom.obj <- function(params, objective) {
  if (!is.null(params[['objective']]) && !is.null(objective))
    stop("Setting objectives in 'params' and 'objective' at the same time is not allowed")

  if (!is.null(objective) && typeof(objective) != 'closure') {
    if (is.character(objective)) {
      msg <- paste(
        "Argument 'objective' is only for custom objectives.",
        "For built-in objectives, pass the objective under 'params'.",
        sep = " "
      )
      error_on_deprecated <- getOption("xgboost.strict_mode", default = FALSE)
      if (error_on_deprecated) {
        stop(msg)
      } else {
        warning(msg, " This warning will become an error in a future version.")
      }
      params$objective <- objective
      return(list(params = params, objective = NULL))
    }
    stop("'objective' must be a function")
  }

  # handle the case when custom objective function was provided through params
  if (!is.null(params[['objective']]) &&
      typeof(params$objective) == 'closure') {
    objective <- params$objective
    params$objective <- NULL
  }
  return(list(params = params, objective = objective))
}

# Performs some checks related to custom evaluation function.
check.custom.eval <- function(params, custom_metric, maximize, early_stopping_rounds, callbacks) {
  if (!is.null(params[['eval_metric']]) && !is.null(custom_metric))
    stop("Setting evaluation metrics in 'params' and 'custom_metric' at the same time is not allowed")

  if (!is.null(custom_metric) && typeof(custom_metric) != 'closure')
    stop("'custom_metric' must be a function")

  # handle a situation when custom eval function was provided through params
  if (!is.null(params[['eval_metric']]) &&
      typeof(params$eval_metric) == 'closure') {
    custom_metric <- params$eval_metric
    params$eval_metric <- NULL
  }

  # require maximize to be set when custom metric and early stopping are used together
  if (!is.null(custom_metric) &&
      is.null(maximize) && (
        !is.null(early_stopping_rounds) ||
        has.callbacks(callbacks, "early_stop")))
    stop("Please set 'maximize' to indicate whether the evaluation metric needs to be maximized or not")

  return(list(params = params, custom_metric = custom_metric))
}


# Update a booster handle for an iteration with dtrain data
xgb.iter.update <- function(bst, dtrain, iter, objective) {
  if (!inherits(dtrain, "xgb.DMatrix")) {
    stop("dtrain must be of xgb.DMatrix class")
  }
  handle <- xgb.get.handle(bst)

  if (is.null(objective)) {
    .Call(XGBoosterUpdateOneIter_R, handle, as.integer(iter), dtrain)
  } else {
    pred <- predict(
      bst,
      dtrain,
      outputmargin = TRUE,
      training = TRUE
    )
    gpair <- objective(pred, dtrain)
    n_samples <- dim(dtrain)[1L]
    grad <- gpair$grad
    hess <- gpair$hess

    if ((is.matrix(grad) && dim(grad)[1L] != n_samples) ||
        (is.vector(grad) && length(grad) != n_samples) ||
        (is.vector(grad) != is.vector(hess))) {
      warning(paste(
        "Since 2.1.0, the shape of the gradient and hessian is required to be ",
        "(n_samples, n_targets) or (n_samples, n_classes). Will reshape assuming ",
        "column-major order.",
        sep = ""
      ))
      grad <- matrix(grad, nrow = n_samples)
      hess <- matrix(hess, nrow = n_samples)
    }

    .Call(
      XGBoosterTrainOneIter_R, handle, dtrain, iter, grad, hess
    )
  }
  return(TRUE)
}


# Evaluate one iteration.
# Returns a named vector of evaluation metrics
# with the names in a 'datasetname-metricname' format.
xgb.iter.eval <- function(bst, evals, iter, custom_metric) {
  handle <- xgb.get.handle(bst)

  if (length(evals) == 0)
    return(NULL)

  evnames <- names(evals)
  if (is.null(custom_metric)) {
    msg <- .Call(XGBoosterEvalOneIter_R, handle, as.integer(iter), evals, as.list(evnames))
    mat <- matrix(strsplit(msg, '\\s+|:')[[1]][-1], nrow = 2)
    res <- structure(as.numeric(mat[2, ]), names = mat[1, ])
  } else {
    res <- sapply(seq_along(evals), function(j) {
      w <- evals[[j]]
      ## predict using all trees
      preds <- predict(bst, w, outputmargin = TRUE, iterationrange = "all")
      eval_res <- custom_metric(preds, w)
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

# Possibly convert the labels into factors, depending on the objective.
# The labels are converted into factors only when the given objective refers to the classification
# or ranking tasks.
convert.labels <- function(labels, objective_name) {
  if (objective_name %in% .CLASSIFICATION_OBJECTIVES()) {
    return(as.factor(labels))
  } else {
    return(labels)
  }
}

# Generates random (stratified if needed) CV folds
generate.cv.folds <- function(nfold, nrows, stratified, label, group, params) {
  if (NROW(group)) {
    if (stratified) {
      warning(
        paste0(
          "Stratified splitting is not supported when using 'group' attribute.",
          " Will use unstratified splitting."
        )
      )
    }
    return(generate.group.folds(nfold, group))
  }
  objective <- params$objective
  if (stratified && !is.character(objective)) {
    warning("Will use unstratified splitting (custom objective used)")
    stratified <- FALSE
  }
  # cannot stratify if label is NULL
  if (stratified && is.null(label)) {
    warning("Will use unstratified splitting (no 'labels' available)")
    stratified <- FALSE
  }

  # cannot do it for rank
  if (is.character(objective) && strtrim(objective, 5) == 'rank:') {
    stop("\n\tAutomatic generation of CV-folds is not implemented for ranking without 'group' field!\n",
         "\tConsider providing pre-computed CV-folds through the 'folds=' parameter.\n")
  }
  # shuffle
  rnd_idx <- sample.int(nrows)
  if (stratified && length(label) == length(rnd_idx)) {
    y <- label[rnd_idx]
    #  - For classification, need to convert y labels to factor before making the folds,
    #    and then do stratification by factor levels.
    #  - For regression, leave y numeric and do stratification by quantiles.
    if (is.character(objective)) {
      y <- convert.labels(y, objective)
    }
    folds <- xgb.createFolds(y = y, k = nfold)
  } else {
    # make simple non-stratified folds
    kstep <- length(rnd_idx) %/% nfold
    folds <- list()
    for (i in seq_len(nfold - 1)) {
      folds[[i]] <- rnd_idx[seq_len(kstep)]
      rnd_idx <- rnd_idx[-seq_len(kstep)]
    }
    folds[[nfold]] <- rnd_idx
  }
  return(folds)
}

generate.group.folds <- function(nfold, group) {
  ngroups <- length(group) - 1
  if (ngroups < nfold) {
    stop("DMatrix has fewer groups than folds.")
  }
  seq_groups <- seq_len(ngroups)
  indices <- lapply(seq_groups, function(gr) seq(group[gr] + 1, group[gr + 1]))
  assignments <- base::split(seq_groups, as.integer(seq_groups %% nfold))
  assignments <- unname(assignments)

  out <- vector("list", nfold)
  randomized_groups <- sample(ngroups)
  for (idx in seq_len(nfold)) {
    groups_idx_test <- randomized_groups[assignments[[idx]]]
    groups_test <- indices[groups_idx_test]
    idx_test <- unlist(groups_test)
    attributes(idx_test)$group_test <- lengths(groups_test)
    attributes(idx_test)$group_train <- lengths(indices[-groups_idx_test])
    out[[idx]] <- idx_test
  }
  return(out)
}

# Creates CV folds stratified by the values of y.
# It was borrowed from caret::createFolds and simplified
# by always returning an unnamed list of fold indices.
xgb.createFolds <- function(y, k) {
  if (is.numeric(y)) {
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

  if (k < length(y)) {
    ## reset levels so that the possible levels and
    ## the levels in the vector are the same
    y <- factor(as.character(y))
    numInClass <- table(y)
    foldVector <- vector(mode = "integer", length(y))

    ## For each class, balance the fold allocation as far
    ## as possible, then resample the remainder.
    ## The final assignment of folds is also randomized.
    for (i in seq_along(numInClass)) {
      ## create a vector of integers from 1:k as many times as possible without
      ## going over the number of samples in the class. Note that if the number
      ## of samples in a class is less than k, nothing is produced here.
      seqVector <- rep(seq_len(k), numInClass[i] %/% k)
      ## add enough random integers to get  length(seqVector) == numInClass[i]
      if (numInClass[i] %% k > 0) seqVector <- c(seqVector, sample.int(k, numInClass[i] %% k))
      ## shuffle the integers for fold assignment and assign to this classes's data
      ## seqVector[sample.int(length(seqVector))] is used to handle length(seqVector) == 1
      foldVector[y == dimnames(numInClass)$y[i]] <- seqVector[sample.int(length(seqVector))]
    }
  } else {
    foldVector <- seq(along = y)
  }

  out <- split(seq(along = y), foldVector)
  names(out) <- NULL
  out
}


#
# Deprectaion notice utilities ------------------------------------------------
#

#' Deprecation notices.
#'
#' At this time, some of the parameter names were changed in order to make the code style more uniform.
#' The deprecated parameters would be removed in the next release.
#'
#' To see all the current deprecated and new parameters, check the `xgboost:::depr_par_lut` table.
#'
#' A deprecation warning is shown when any of the deprecated parameters is used in a call.
#' An additional warning is shown when there was a partial match to a deprecated parameter
#' (as R is able to partially match parameter names).
#'
#' @name xgboost-deprecated
NULL

#' Model Serialization and Compatibility
#'
#' @description
#' When it comes to serializing XGBoost models, it's possible to use R serializers such as
#' [save()] or [saveRDS()] to serialize an XGBoost model object, but XGBoost also provides
#' its own serializers with better compatibility guarantees, which allow loading
#' said models in other language bindings of XGBoost.
#'
#' Note that an `xgb.Booster` object (**as produced by [xgb.train()]**, see rest of the doc
#' for objects produced by [xgboost()]), outside of its core components, might also keep:
#' - Additional model configuration (accessible through [xgb.config()]), which includes
#'   model fitting parameters like `max_depth` and runtime parameters like `nthread`.
#'   These are not necessarily useful for prediction/importance/plotting.
#' - Additional R specific attributes  - e.g. results of callbacks, such as evaluation logs,
#'   which are kept as a `data.table` object, accessible through
#'   `attributes(model)$evaluation_log` if present.
#'
#' The first one (configurations) does not have the same compatibility guarantees as
#' the model itself, including attributes that are set and accessed through
#' [xgb.attributes()] - that is, such configuration might be lost after loading the
#' booster in a different XGBoost version, regardless of the serializer that was used.
#' These are saved when using [saveRDS()], but will be discarded if loaded into an
#' incompatible XGBoost version. They are not saved when using XGBoost's
#' serializers from its public interface including [xgb.save()] and [xgb.save.raw()].
#'
#' The second ones (R attributes) are not part of the standard XGBoost model structure,
#' and thus are not saved when using XGBoost's own serializers. These attributes are
#' only used for informational purposes, such as keeping track of evaluation metrics as
#' the model was fit, or saving the R call that produced the model, but are otherwise
#' not used for prediction / importance / plotting / etc.
#' These R attributes are only preserved when using R's serializers.
#'
#' In addition to the regular `xgb.Booster` objects produced by [xgb.train()], the
#' function [xgboost()] produces objects with a different subclass `xgboost` (which
#' inherits from `xgb.Booster`), which keeps other additional metadata as R attributes
#' such as class names in classification problems, and which has a dedicated `predict`
#' method that uses different defaults and takes different argument names. XGBoost's
#' own serializers can work with this `xgboost` class, but as they do not keep R
#' attributes, the resulting object, when deserialized, is downcasted to the regular
#' `xgb.Booster` class (i.e. it loses the metadata, and the resulting object will use
#' [predict.xgb.Booster()] instead of [predict.xgboost()]) - for these `xgboost` objects,
#' `saveRDS` might thus be a better option if the extra functionalities are needed.
#'
#' Note that XGBoost models in R starting from version `2.1.0` and onwards, and
#' XGBoost models before version `2.1.0`; have a very different R object structure and
#' are incompatible with each other. Hence, models that were saved with R serializers
#' like [saveRDS()] or [save()] before version `2.1.0` will not work with latter
#' `xgboost` versions and vice versa. Be aware that the structure of R model objects
#' could in theory change again in the future, so XGBoost's serializers should be
#' preferred for long-term storage.
#'
#' Furthermore, note that model objects from XGBoost might not be serializable with third-party
#' R packages like `qs` or `qs2`.
#'
#' @details
#' Use [xgb.save()] to save the XGBoost model as a stand-alone file. You may opt into
#' the JSON format by specifying the JSON extension. To read the model back, use
#' [xgb.load()].
#'
#' Use [xgb.save.raw()] to save the XGBoost model as a sequence (vector) of raw bytes
#' in a future-proof manner. Future releases of XGBoost will be able to read the raw bytes and
#' re-construct the corresponding model. To read the model back, use [xgb.load.raw()].
#' The [xgb.save.raw()] function is useful if you would like to persist the XGBoost model
#' as part of another R object.
#'
#' Use [saveRDS()] if you require the R-specific attributes that a booster might have, such
#' as evaluation logs or the model class `xgboost` instead of `xgb.Booster`, but note that
#' future compatibility of such objects is outside XGBoost's control as it relies on R's
#' serialization format (see e.g. the details section in [serialize] and [save()] from base R).
#'
#' For more details and explanation about model persistence and archival, consult the page
#' \url{https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html}.
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(agaricus.train$data, label = agaricus.train$label),
#'   nrounds = 2,
#'   params = xgb.params(
#'     max_depth = 2,
#'     nthread = 2,
#'     objective = "binary:logistic"
#'   )
#' )
#'
#' # Save as a stand-alone file; load it with xgb.load()
#' fname <- file.path(tempdir(), "xgb_model.ubj")
#' xgb.save(bst, fname)
#' bst2 <- xgb.load(fname)
#'
#' # Save as a stand-alone file (JSON); load it with xgb.load()
#' fname <- file.path(tempdir(), "xgb_model.json")
#' xgb.save(bst, fname)
#' bst2 <- xgb.load(fname)
#'
#' # Save as a raw byte vector; load it with xgb.load.raw()
#' xgb_bytes <- xgb.save.raw(bst)
#' bst2 <- xgb.load.raw(xgb_bytes)
#'
#' # Persist XGBoost model as part of another R object
#' obj <- list(xgb_model_bytes = xgb.save.raw(bst), description = "My first XGBoost model")
#' # Persist the R object. Here, saveRDS() is okay, since it doesn't persist
#' # xgb.Booster directly. What's being persisted is the future-proof byte representation
#' # as given by xgb.save.raw().
#' fname <- file.path(tempdir(), "my_object.Rds")
#' saveRDS(obj, fname)
#' # Read back the R object
#' obj2 <- readRDS(fname)
#' # Re-construct xgb.Booster object from the bytes
#' bst2 <- xgb.load.raw(obj2$xgb_model_bytes)
#'
#' @name a-compatibility-note-for-saveRDS-save
NULL

#' @name xgboost-options
#' @title XGBoost Options
#' @description XGBoost offers an \link[base:options]{option setting} for controlling the behavior
#' of deprecated and removed function arguments.
#'
#' Some of the arguments in functions like [xgb.train()] or [predict.xgb.Booster()] been renamed
#' from how they were in previous versions, or have been removed.
#'
#' In order to make the transition to newer XGBoost versions easier, some of these parameters are
#' still accepted but issue a warning when using them. \bold{Note that these warnings will become
#' errors in the future!!} - this is just a temporary workaround to make the transition easier.
#'
#' One can optionally use 'strict mode' to turn these warnings into errors, in order to ensure
#' that code calling xgboost will still work once those are removed in future releases.
#'
#' Currently, the only supported option is `xgboost.strict_mode`, which can be set to `TRUE` or
#' `FALSE` (default).
#' @examples
#' options("xgboost.strict_mode" = FALSE)
#' options("xgboost.strict_mode" = TRUE)
NULL

# Lookup table for the deprecated parameters bookkeeping
deprecated_train_params <- list(
  renamed = list(
    'print.every.n' = 'print_every_n',
    'early.stop.round' = 'early_stopping_rounds',
    'training.data' = 'data',
    'dtrain' = 'data',
    'watchlist' = 'evals',
    'feval' = 'custom_metric'
  ),
  removed = character()
)
deprecated_xgboost_params <- list(
  renamed = list(
    'data' = 'x',
    'label' = 'y',
    'eta' = 'learning_rate',
    'gamma' = 'min_split_loss',
    'lambda' = 'reg_lambda',
    'alpha' = 'reg_alpha',
    'min.split.loss' = 'min_split_loss',
    'reg.lambda' = 'reg_lambda',
    'reg.alpha' = 'reg_alpha',
    'watchlist' = 'evals'
  ),
  removed = c(
    'params',
    'save_period',
    'save_name',
    'xgb_model',
    'callbacks',
    'missing',
    'maximize'
  )
)
deprecated_dttree_params <- list(
  renamed = list('n_first_tree' = 'trees'),
  removed = c("feature_names", "text")
)
deprecated_plotimp_params <- list(
  renamed = list(
    'plot.height' = 'plot_height',
    'plot.width' = 'plot_width'
  ),
  removed = character()
)
deprecated_multitrees_params <- list(
  renamed = c(
    deprecated_plotimp_params$renamed,
    list('features.keep' = 'features_keep')
  ),
  removed = "feature_names"
)
deprecated_dump_params <- list(
  renamed = list('with.stats' = 'with_stats'),
  removed = character()
)
deprecated_plottree_params <- c(
  renamed = list(
    deprecated_plotimp_params$renamed,
    deprecated_dump_params$renamed,
    list('trees' = 'tree_idx')
  ),
  removed = c("show_node_id", "feature_names")
)
deprecated_predict_params <- list(
  renamed = list("ntreelimit" = "iterationrange"),
  removed = "reshape"
)
deprecated_dmatrix_params <- list(
  renamed = character(),
  removed = "info"
)

# These got moved from 'info' to function arguments
args_previous_dmatrix_info <- c("label", "weight", "base_margin", "group")

# Checks the dot-parameters for deprecated names
# (including partial matching), gives a deprecation warning,
# and sets new parameters to the old parameters' values within its parent frame.
# WARNING: has side-effects
check.deprecation <- function(
  deprecated_list,
  fn_call,
  ...,
  env = parent.frame(),
  allow_unrecognized = FALSE
) {
  params <- list(...)
  if (length(params) == 0) {
    return(NULL)
  }
  error_on_deprecated <- getOption("xgboost.strict_mode", default = FALSE)
  throw_err_or_depr_msg <- function(...) {
    if (error_on_deprecated) {
      stop(...)
    } else {
      warning(..., " This warning will become an error in a future version.")
    }
  }

  if (is.null(names(params)) || min(nchar(names(params))) == 0L) {
    throw_err_or_depr_msg("Passed invalid positional arguments")
  }
  list_renamed <- deprecated_list$renamed
  list_removed <- deprecated_list$removed
  has_params_arg <-
    length(list_renamed) == length(deprecated_train_params$renamed) &&
    list_renamed[[1L]] == deprecated_train_params$renamed[[1L]]
  is_dmatrix_constructor <-
    length(list_removed) == length(deprecated_dmatrix_params$removed) &&
    list_removed[[1L]] == deprecated_dmatrix_params$removed[[1L]]
  all_match <- pmatch(names(params), names(list_renamed))
  # throw error on unrecognized parameters
  if (!allow_unrecognized && anyNA(all_match)) {

    names_unrecognized <- names(params)[is.na(all_match)]
    # make it informative if they match something that goes under 'params'
    if (has_params_arg) {
      names_params <- formalArgs(xgb.params)
      names_params <- c(names_params, gsub("_", ".", names_params, fixed = TRUE))
      names_under_params <- intersect(names_unrecognized, names_params)
      if (length(names_under_params)) {
        if (error_on_deprecated) {
          stop(
            "Passed invalid function arguments: ",
            paste(head(names_under_params), collapse = ", "),
            ". These should be passed as a list to argument 'params'."
          )
        } else {
          warning(
            "Passed invalid function arguments: ",
            paste(head(names_under_params), collapse = ", "),
            ". These should be passed as a list to argument 'params'.",
            " Conversion from argument to 'params' entry will be done automatically, but this ",
            "behavior will become an error in a future version."
          )
          if (any(names_under_params %in% names(env[["params"]]))) {
            repeteated_params <- intersect(names_under_params, names(env[["params"]]))
            stop(
              "Passed entries as both function argument(s) and as elements under 'params': ",
              paste(head(repeteated_params), collapse = ", ")
            )
          } else {
            env[["params"]] <- c(env[["params"]], params[names_under_params])
          }
        }
        names_unrecognized <- setdiff(names_unrecognized, names_under_params)
      }
    } else if (is_dmatrix_constructor && NROW(params$info)) {
      # same thing for the earlier 'info' in 'xgb.DMatrix'
      throw_err_or_depr_msg(
        "Passed invalid argument 'info' - entries on it should be passed as direct arguments."
      )
      entries_info <- names(params$info)
      if (length(setdiff(entries_info, args_previous_dmatrix_info))) {
        stop(
          "Passed unrecognized entries under info: ",
          paste(setdiff(entries_info, args_previous_dmatrix_info) |> head(), collapse = ", ")
        )
      }
      for (entry_name in entries_info) {
        if (!is.null(env[[entry_name]])) {
          stop("Passed entry under both 'info' and function argument(s): ", entry_name)
        }
        env[[entry_name]] <- params$info[[entry_name]]
      }
      names_unrecognized <- setdiff(names_unrecognized, "info")
    }

    # check for parameters that were removed from a previous version
    names_removed <- intersect(names_unrecognized, list_removed)
    if (length(names_removed)) {
      throw_err_or_depr_msg(
        "Parameter(s) have been removed from this function: ",
        paste(names_removed, collapse = ", "), "."
      )
      names_unrecognized <- setdiff(names_unrecognized, list_removed)
    }

    # otherwise throw a generic error
    if (length(names_unrecognized)) {
      throw_err_or_depr_msg(
        "Passed unrecognized parameters: ",
        paste(head(names_unrecognized), collapse = ", "), "."
      )
    }

  } else {

    names_removed <- intersect(names(params)[is.na(all_match)], list_removed)
    if (length(names_removed)) {
      throw_err_or_depr_msg(
        "Parameter(s) have been removed from this function: ",
        paste(names_removed, collapse = ", "), "."
      )
    }

  }

  matched_params <- list_renamed[all_match[!is.na(all_match)]]
  idx_orig <- seq_along(params)[!is.na(all_match)]
  function_args_passed <- names(as.list(fn_call))[-1L]
  for (idx in seq_along(matched_params)) {
    match_old <- names(matched_params)[[idx]]
    match_new <- matched_params[[idx]]
    throw_err_or_depr_msg(
      "Parameter '", match_old, "' has been renamed to '",
      match_new, "'."
    )
    if (match_new %in% function_args_passed) {
      stop("Passed both '", match_new, "' and '", match_old, "'.")
    }
    env[[match_new]] <- params[[idx_orig[idx]]]
  }
}
