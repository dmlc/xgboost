prescreen.parameters <- function(params) {
  if (!NROW(params)) {
    return(list())
  }
  if (!is.list(params)) {
    stop("'params' must be a list or NULL.")
  }

  params <- params[!is.null(params)]

  if ("num_class" %in% names(params)) {
    stop("'num_class' cannot be manually specified for 'xgboost()'. Pass a factor 'y' instead.")
  }
  if ("process_type" %in% names(params)) {
    if (params$process_type != "default") {
      stop("Non-default 'process_type' is not supported for 'xgboost()'. Try 'xgb.train()'.")
    }
  }

  return(params)
}

prescreen.objective <- function(objective) {
  if (!is.null(objective)) {
    if (objective %in% .OBJECTIVES_NON_DEFAULT_MODE()) {
      stop(
        "Objectives with non-default prediction mode (",
        paste(.OBJECTIVES_NON_DEFAULT_MODE(), collapse = ", "),
        ") are not supported in 'xgboost()'. Try 'xgb.train()'."
      )
    }

    if (!is.character(objective) || length(objective) != 1L || is.na(objective)) {
      stop("'objective' must be a single character/string variable.")
    }
  }
}

process.base.margin <- function(base_margin, nrows, ncols) {
  if (!NROW(base_margin)) {
    return(NULL)
  }
  if (is.array(base_margin) && length(dim(base_margin)) > 2) {
    stop(
      "'base_margin' should not have more than 2 dimensions for any objective (got: ",
      length(dim(base_margin)),
      " dimensions)."
    )
  }
  if (inherits(base_margin, c("sparseMatrix", "sparseVector"))) {
    warning(
      "Got a sparse matrix type (class: ",
      paste(class(base_margin), collapse = ", "),
      ") for 'base_margin'. Will convert to dense matrix."
    )
    base_margin <- as.matrix(base_margin)
  }
  if (NROW(base_margin) != nrows) {
    stop(
      "'base_margin' has incorrect number of rows. Expected: ",
      nrows,
      ". Got: ",
      NROW(base_margin)
    )
  }

  if (ncols == 1L) {
    if (inherits(base_margin, c("matrix", "data.frame"))) {
      if (ncol(base_margin) != 1L) {
        stop("'base_margin' should be a 1-d vector for the given objective and data.")
      }
      if (is.data.frame(base_margin)) {
        base_margin <- base_margin[[1L]]
      } else {
        base_margin <- base_margin[, 1L]
      }
    }
    if (!is.numeric(base_margin)) {
      base_margin <- as.numeric(base_margin)
    }
  } else {
    supported_multicol <- c("matrix", "data.frame")
    if (!inherits(base_margin, supported_multicol)) {
      stop(
        "'base_margin' should be a matrix with ",
        ncols,
        " columns for the given objective and data. Got class: ",
        paste(class(base_margin), collapse = ", ")
      )
    }
    if (ncol(base_margin) != ncols) {
      stop(
        "'base_margin' has incorrect number of columns. Expected: ",
        ncols,
        ". Got: ",
        ncol(base_margin)
      )
    }
    if (!is.matrix(base_margin)) {
      base_margin <- as.matrix(base_margin)
    }
  }

  return(base_margin)
}

process.y.margin.and.objective <- function(
  y,
  base_margin,
  objective,
  params
) {

  if (!NROW(y)) {
    stop("Passed empty 'y'.")
  }

  if (is.array(y) && length(dim(y)) > 2) {
    stop(
      "'y' should not have more than 2 dimensions for any objective (got: ",
      length(dim(y)),
      ")."
    )
  }

  if (inherits(y, c("sparseMatrix", "sparseVector"))) {
    warning(
      "Got a sparse matrix type (class: ",
      paste(class(y), collapse = ", "),
      ") for 'y'. Will convert to dense matrix."
    )
    y <- as.matrix(y)
  }

  if (is.character(y)) {
    if (!is.vector(y)) {
      if (NCOL(y) > 1L) {
        stop("Multi-column categorical 'y' is not supported.")
      }
      y <- as.vector(y)
    }
    y <- factor(y)
  }

  if (is.logical(y)) {
    if (!is.vector(y)) {
      if (NCOL(y) > 1L) {
        stop("Multi-column logical/boolean 'y' is not supported.")
      }
      y <- as.vector(y)
    }
    y <- factor(y, c(FALSE, TRUE))
  }

  if (is.factor(y)) {

    y_levels <- levels(y)
    if (length(y_levels) < 2) {
      stop("Factor 'y' has less than 2 levels.")
    }
    if (length(y_levels) == 2) {
      if (is.null(objective)) {
        objective <- "binary:logistic"
      } else {
        if (!(objective %in% .BINARY_CLASSIF_OBJECTIVES())) {
          stop(
            "Got binary 'y' - supported objectives for this data are: ",
            paste(.BINARY_CLASSIF_OBJECTIVES(), collapse = ", "),
            ". Was passed: ",
            objective
          )
        }
      }

      if (!is.null(base_margin)) {
        base_margin <- process.base.margin(base_margin, length(y), 1)
      }

      out <- list(
        params = list(
          objective = objective
        ),
        metadata = list(
          y_levels = y_levels,
          n_targets = 1
        )
      )
    } else { # length(levels) > 2
      if (is.null(objective)) {
        objective <- "multi:softprob"
      } else {
        if (!(objective %in% .MULTICLASS_CLASSIF_OBJECTIVES())) {
          stop(
            "Got non-binary factor 'y' - supported objectives for this data are: ",
            paste(.MULTICLASS_CLASSIF_OBJECTIVES(), collapse = ", "),
            ". Was passed: ",
            objective
          )
        }
      }

      if (!is.null(base_margin)) {
        base_margin <- process.base.margin(base_margin, length(y), length(y_levels))
      }

      out <- list(
        params = list(
          objective = objective,
          num_class = length(y_levels)
        ),
        metadata = list(
          y_levels = y_levels,
          n_targets = length(y_levels)
        )
      )
    }

    out$dmatrix_args <- list(
      label = as.numeric(y) - 1,
      base_margin = base_margin
    )

  } else if (inherits(y, "Surv")) {

    y_attr <- attributes(y)
    supported_surv_types <- c("left", "right", "interval")
    if (!(y_attr$type %in% supported_surv_types)) {
      stop(
        "Survival objectives are only supported for types: ",
        paste(supported_surv_types, collapse = ", "),
        ". Was passed: ",
        y_attr$type
      )
    }

    if (is.null(objective)) {
      objective <- "survival:aft"
    } else {
      if (y_attr$type == "right") {
        if (!(objective %in% .SURVIVAL_RIGHT_CENSORING_OBJECTIVES())) {
          stop(
            "Got right-censored 'y' variable - supported objectives for this data are: ",
            paste(.SURVIVAL_RIGHT_CENSORING_OBJECTIVES(), collapse = ", "),
            ". Was passed: ",
            objective
          )
        }
      } else {
        if (!(objective %in% .SURVIVAL_ALL_CENSORING_OBJECTIVES())) {
          stop(
            "Got ", y_attr$type, "-censored 'y' variable - supported objectives for this data are:",
            paste(.SURVIVAL_ALL_CENSORING_OBJECTIVES(), collapse = ", "),
            ". Was passed: ",
            objective
          )
        }
      }
    }

    if (!is.null(base_margin)) {
      base_margin <- process.base.margin(base_margin, nrow(y), 1)
    }

    out <- list(
      params = list(
        objective = objective
      ),
      metadata = list(
        n_targets = 1
      )
    )

    # Note: the 'Surv' object class that is passed as 'y' might have either 2 or 3 columns
    # depending on the type of censoring, and the last column in both cases is the one that
    # indicates the observation type (e.g. censored / uncensored).
    # In the case of interval censoring, the second column will not always have values with
    # infinites filled in. For more information, see the code behind the 'print.Surv' method.

    if (objective == "survival:cox") {
      # Can only get here when using right censoring
      if (y_attr$type != "right") {
        stop("Internal error.")
      }

      out$dmatrix_args <- list(
        label = y[, 1L] * (2 * (y[, 2L] - 0.5))
      )

    } else {
      if (y_attr$type == "left") {
        lb <- ifelse(
          y[, 2L] == 0,
          0,
          y[, 1L]
        )
        ub <- y[, 1L]
        out$dmatrix_args <- list(
          label_lower_bound = lb,
          label_upper_bound = ub
        )
      } else if (y_attr$type == "right") {
        lb <- y[, 1L]
        ub <- ifelse(
          y[, 2L] == 0,
          Inf,
          y[, 1L]
        )
        out$dmatrix_args <- list(
          label_lower_bound = lb,
          label_upper_bound = ub
        )
      } else if (y_attr$type == "interval") {
        out$dmatrix_args <- list(
          label_lower_bound = ifelse(y[, 3L] == 2, 0, y[, 1L]),
          label_upper_bound = ifelse(
            y[, 3L] == 0, Inf,
            ifelse(y[, 3L] == 3, y[, 2L], y[, 1L])
          )
        )
      }

      if (min(out$dmatrix_args$label_lower_bound) < 0) {
        stop("Survival objectives are only defined for non-negative 'y'.")
      }
    }

    out$dmatrix_args$base_margin <- base_margin

  } else if (is.vector(y)) {

    if (is.null(objective)) {
      objective <- "reg:squarederror"
    } else if (!(objective %in% .REGRESSION_OBJECTIVES())) {
      stop(
        "Got numeric 'y' - supported objectives for this data are: ",
        paste(.REGRESSION_OBJECTIVES(), collapse = ", "),
        ". Was passed: ",
        objective
      )
    }

    n_targets <- 1L
    if (objective == "reg:quantileerror" && NROW(params$quantile_alpha) > 1) {
      n_targets <- NROW(params$quantile_alpha)
    }

    if (!is.null(base_margin)) {
      base_margin <- process.base.margin(base_margin, length(y), n_targets)
    }

    out <- list(
      params = list(
        objective = objective
      ),
      metadata = list(
        n_targets = n_targets
      ),
      dmatrix_args = list(
        label = as.numeric(y),
        base_margin = base_margin
      )
    )

  } else if (is.data.frame(y)) {
    if (ncol(y) == 1L) {
      return(process.y.margin.and.objective(y[[1L]], base_margin, objective, params))
    }

    if (is.null(objective)) {
      objective <- "reg:squarederror"
    } else if (!(objective %in% .MULTI_TARGET_OBJECTIVES())) {
      stop(
        "Got multi-column 'y' - supported objectives for this data are: ",
        paste(.MULTI_TARGET_OBJECTIVES(), collapse = ", "),
        ". Was passed: ",
        objective
      )
    }

    y_names <- names(y)
    y <- lapply(y, function(x) {
      if (!inherits(x, c("numeric", "integer"))) {
        stop(
          "Multi-target 'y' only supports 'numeric' and 'integer' types. Got: ",
          paste(class(x), collapse = ", ")
        )
      }
      return(as.numeric(x))
    })
    y <- as.data.frame(y) |> as.matrix()

    if (!is.null(base_margin)) {
      base_margin <- process.base.margin(base_margin, length(y), ncol(y))
    }

    out <- list(
      params = list(
        objective = objective
      ),
      dmatrix_args = list(
        label = y,
        base_margin = base_margin
      ),
      metadata = list(
        y_names = y_names,
        n_targets = ncol(y)
      )
    )

  } else if (is.matrix(y)) {
    if (ncol(y) == 1L) {
      return(process.y.margin.and.objective(as.vector(y), base_margin, objective, params))
    }

    if (!is.null(objective) && !(objective %in% .MULTI_TARGET_OBJECTIVES())) {
      stop(
        "Got multi-column 'y' - supported objectives for this data are: ",
        paste(.MULTI_TARGET_OBJECTIVES(), collapse = ", "),
        ". Was passed: ",
        objective
      )
    }
    if (is.null(objective)) {
      objective <- "reg:squarederror"
    }

    y_names <- colnames(y)
    if (storage.mode(y) != "double") {
      storage.mode(y) <- "double"
    }

    if (!is.null(base_margin)) {
      base_margin <- process.base.margin(base_margin, nrow(y), ncol(y))
    }

    out <- list(
      params = list(
        objective = objective
      ),
      dmatrix_args = list(
        label = y,
        base_margin = base_margin
      ),
      metadata = list(
        n_targets = ncol(y)
      )
    )

    if (NROW(y_names) == ncol(y)) {
      out$metadata$y_names <- y_names
    }

  } else {
    stop("Passed 'y' object with unsupported class: ", paste(class(y), collapse = ", "))
  }

  return(out)
}

process.row.weights <- function(w, lst_args) {
  if (!is.null(w)) {
    if ("label" %in% names(lst_args$dmatrix_args)) {
      nrow_y <- NROW(lst_args$dmatrix_args$label)
    } else if ("label_lower_bound" %in% names(lst_args$dmatrix_args)) {
      nrow_y <- length(lst_args$dmatrix_args$label_lower_bound)
    } else {
      stop("Internal error.")
    }
    if (!is.numeric(w)) {
      w <- as.numeric(w)
    }
    if (length(w) != nrow_y) {
      stop(
        "'weights' must be a 1-d vector with the same length as 'y' (",
        length(w), " vs. ", nrow_y, ")."
      )
    }
    lst_args$dmatrix_args$weight <- w
  }
  return(lst_args)
}

check.nthreads <- function(nthreads) {
  if (is.null(nthreads)) {
    return(1L)
  }
  if (!inherits(nthreads, c("numeric", "integer")) || !NROW(nthreads)) {
    stop("'nthreads' must be a positive scalar value.")
  }
  if (length(nthreads) > 1L) {
    nthreads <- utils::head(nthreads, 1L)
  }
  if (is.na(nthreads) || nthreads < 0) {
    stop("Passed invalid 'nthreads': ", nthreads)
  }
  if (is.numeric(nthreads)) {
    if (floor(nthreads) != nthreads) {
      stop("'nthreads' must be an integer.")
    }
  }
  return(as.integer(nthreads))
}

check.can.use.qdm <- function(x, params) {
  if ("booster" %in% names(params)) {
    if (params$booster == "gblinear") {
      return(FALSE)
    }
  }
  if ("tree_method" %in% names(params)) {
    if (params$tree_method %in% c("exact", "approx")) {
      return(FALSE)
    }
  }
  return(TRUE)
}

process.x.and.col.args <- function(
  x,
  monotone_constraints,
  interaction_constraints,
  feature_weights,
  lst_args,
  use_qdm
) {
  if (is.null(x)) {
    stop("'x' cannot be NULL.")
  }
  if (inherits(x, "xgb.DMatrix")) {
    stop("Cannot pass 'xgb.DMatrix' as 'x' to 'xgboost()'. Try 'xgb.train()' instead.")
  }
  supported_x_types <- c("data.frame", "matrix", "dgTMatrix", "dgCMatrix", "dgRMatrix")
  if (!inherits(x, supported_x_types)) {
    stop(
      "'x' must be one of the following classes: ",
      paste(supported_x_types, collapse = ", "),
      ". Got: ",
      paste(class(x), collapse = ", ")
    )
  }
  if (use_qdm && inherits(x, "sparseMatrix") && !inherits(x, "dgRMatrix")) {
    x <- methods::as(x, "RsparseMatrix")
    if (!inherits(x, "RsparseMatrix")) {
      stop("Internal error: casting sparse matrix did not yield 'dgRMatrix'.")
    }
  }

  if (NROW(feature_weights)) {
    if (is.list(feature_weights)) {
      feature_weights <- unlist(feature_weights)
    }
    if (!inherits(feature_weights, c("numeric", "integer"))) {
      stop("'feature_weights' must be a numeric vector or named list matching to columns of 'x'.")
    }
    if (NROW(names(feature_weights)) && NROW(colnames(x))) {
      matched <- match(colnames(x), names(feature_weights))
      matched <- matched[!is.na(matched)]
      matched <- matched[!duplicated(matched)]
      if (length(matched) > 0 && length(matched) < length(feature_weights)) {
        stop(
          "'feature_weights' names do not contain all columns of 'x'. Missing: ",
          utils::head(setdiff(colnames(x), names(feature_weights)))
        )
      }
      if (length(matched)) {
        feature_weights <- feature_weights[matched]
      } else {
        warning("Names of 'feature_weights' do not match with 'x'. Names will be ignored.")
      }
    }

    lst_args$dmatrix_args$feature_weights <- unname(feature_weights)
  }

  if (NROW(monotone_constraints)) {

    if (NROW(monotone_constraints) > ncol(x)) {
      stop(
        "'monotone_constraints' contains more entries than there are columns in 'x' (",
        NROW(monotone_constraints), " vs. ", ncol(x), ")."
      )
    }

    if (is.list(monotone_constraints)) {

      if (!NROW(names(monotone_constraints))) {
        stop(
          "If passing 'monotone_constraints' as a named list,",
          " must have names matching to columns of 'x'."
        )
      }
      if (!NROW(colnames(x))) {
        stop("If passing 'monotone_constraints' as a named list, 'x' must have column names.")
      }
      if (anyDuplicated(names(monotone_constraints))) {
        stop(
          "'monotone_constraints' contains duplicated names: ",
          paste(
            names(monotone_constraints)[duplicated(names(monotone_constraints))] |> utils::head(),
            collapse = ", "
          )
        )
      }
      if (NROW(setdiff(names(monotone_constraints), colnames(x)))) {
        stop(
          "'monotone_constraints' contains column names not present in 'x': ",
          paste(utils::head(names(monotone_constraints)), collapse = ", ")
        )
      }

      vec_monotone_constr <- rep(0, ncol(x))
      matched <- match(names(monotone_constraints), colnames(x))
      vec_monotone_constr[matched] <- unlist(monotone_constraints)
      lst_args$params$monotone_constraints <- unname(vec_monotone_constr)

    } else if (inherits(monotone_constraints, c("numeric", "integer"))) {

      if (NROW(names(monotone_constraints)) && NROW(colnames(x))) {
        if (length(monotone_constraints) < ncol(x)) {
          return(
            process.x.and.col.args(
              x,
              as.list(monotone_constraints),
              interaction_constraints,
              feature_weights,
              lst_args,
              use_qdm
            )
          )
        } else {
          matched <- match(names(monotone_constraints), colnames(x))
          matched <- matched[!is.na(matched)]
          matched <- matched[!duplicated(matched)]
          if (length(matched)) {
            monotone_constraints <- monotone_constraints[matched]
          } else {
            warning("Names of 'monotone_constraints' do not match with 'x'. Names will be ignored.")
          }
        }
      } else {
        if (length(monotone_constraints) != ncol(x)) {
          stop(
            "If passing 'monotone_constraints' as unnamed vector or not using column names,",
            " must have length matching to number of columns in 'x'. Got: ",
            length(monotone_constraints), " (vs. ", ncol(x), ")"
          )
        }
      }

      lst_args$params$monotone_constraints <- unname(monotone_constraints)

    } else if (is.character(monotone_constraints)) {
      lst_args$params$monotone_constraints <- monotone_constraints
    } else {
      stop(
        "Passed unsupported type for 'monotone_constraints': ",
        paste(class(monotone_constraints), collapse = ", ")
      )
    }
  }

  if (NROW(interaction_constraints)) {
    if (!is.list(interaction_constraints)) {
      stop("'interaction_constraints' must be a list of vectors.")
    }
    cnames <- colnames(x)
    lst_args$params$interaction_constraints <- lapply(interaction_constraints, function(idx) {
      if (!NROW(idx)) {
        stop("Elements in 'interaction_constraints' cannot be empty.")
      }

      if (is.character(idx)) {
        if (!NROW(cnames)) {
          stop(
            "Passed a character vector for 'interaction_constraints', but 'x' ",
            "has no column names to match them against."
          )
        }
        out <- match(idx, cnames) - 1L
        if (anyNA(out)) {
          stop(
            "'interaction_constraints' contains column names not present in 'x': ",
            paste(utils::head(idx[which(is.na(out))]), collapse = ", ")
          )
        }
        return(out)
      } else if (inherits(idx, c("numeric", "integer"))) {
        if (anyNA(idx)) {
          stop("'interaction_constraints' cannot contain NA values.")
        }
        if (min(idx) < 1) {
          stop("Column indices for 'interaction_constraints' must follow base-1 indexing.")
        }
        if (max(idx) > ncol(x)) {
          stop("'interaction_constraints' contains invalid column indices.")
        }
        if (is.numeric(idx)) {
          if (any(idx != floor(idx))) {
            stop(
              "'interaction_constraints' must contain only integer indices. Got non-integer: ",
              paste(utils::head(idx[which(idx != floor(idx))]), collapse = ", ")
            )
          }
        }
        return(idx - 1L)
      } else {
        stop(
          "Elements in 'interaction_constraints' must be vectors of types ",
          "'integer', 'numeric', or 'character'. Got: ",
          paste(class(idx), collapse = ", ")
        )
      }
    })
  }

  lst_args$dmatrix_args$data <- x
  return(lst_args)
}

#' Fit XGBoost Model
#'
#' @export
#' @description
#' Fits an XGBoost model (boosted decision tree ensemble) to given x/y data.
#'
#' See the tutorial [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
#' for a longer explanation of what XGBoost does.
#'
#' This function is intended to provide a more user-friendly interface for XGBoost that follows
#' R's conventions for model fitting and predictions, but which doesn't expose all of the
#' possible functionalities of the core XGBoost library.
#'
#' See [xgb.train()] for a more flexible low-level alternative which is similar across different
#' language bindings of XGBoost and which exposes the full library's functionalities.
#'
#' @details
#' For package authors using 'xgboost' as a dependency, it is highly recommended to use
#' [xgb.train()] in package code instead of [xgboost()], since it has a more stable interface
#' and performs fewer data conversions and copies along the way.
#' @references
#'   - Chen, Tianqi, and Carlos Guestrin. "Xgboost: A scalable tree boosting system."
#'     Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and
#'     data mining. 2016.
#'   - \url{https://xgboost.readthedocs.io/en/stable/}
#' @param x The features / covariates. Can be passed as:
#'   - A numeric or integer `matrix`.
#'   - A `data.frame`, in which all columns are one of the following types:
#'     - `numeric`
#'     - `integer`
#'     - `logical`
#'     - `factor`
#'
#'     Columns of `factor` type will be assumed to be categorical, while other column types will
#'     be assumed to be numeric.
#'   - A sparse matrix from the `Matrix` package, either as `dgCMatrix` or `dgRMatrix` class.
#'
#'   Note that categorical features are only supported for `data.frame` inputs, and are automatically
#'   determined based on their types. See [xgb.train()] with [xgb.DMatrix()] for more flexible
#'   variants that would allow something like categorical features on sparse matrices.
#' @param y The response variable. Allowed values are:
#'   - A numeric or integer vector (for regression tasks).
#'   - A factor or character vector (for binary and multi-class classification tasks).
#'   - A logical (boolean) vector (for binary classification tasks).
#'   - A numeric or integer matrix or `data.frame` with numeric/integer columns
#'     (for multi-task regression tasks).
#'   - A `Surv` object from the 'survival' package (for survival tasks).
#'
#'   If `objective` is `NULL`, the right task will be determined automatically based on
#'   the class of `y`.
#'
#'   If `objective` is not `NULL`, it must match with the type of `y` - e.g. `factor` types of `y`
#'   can only be used with classification objectives and vice-versa.
#'
#'   For binary classification, the last factor level of `y` will be used as the "positive"
#'   class - that is, the numbers from `predict` will reflect the probabilities of belonging to this
#'   class instead of to the first factor level. If `y` is a `logical` vector, then `TRUE` will be
#'   set as the last level.
#' @param objective Optimization objective to minimize based on the supplied data, to be passed
#'   by name as a string / character (e.g. `reg:absoluteerror`). See the
#'   [Learning Task Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters)
#'   page for more detailed information on allowed values.
#'
#'   If `NULL` (the default), will be automatically determined from `y` according to the following
#'   logic:
#'   - If `y` is a factor with 2 levels, will use `binary:logistic`.
#'   - If `y` is a factor with more than 2 levels, will use `multi:softprob` (number of classes
#'     will be determined automatically, should not be passed under `params`).
#'   - If `y` is a `Surv` object from the `survival` package, will use `survival:aft` (note that
#'     the only types supported are left / right / interval censored).
#'   - Otherwise, will use `reg:squarederror`.
#'
#'   If `objective` is not `NULL`, it must match with the type of `y` - e.g. `factor` types of `y`
#'   can only be used with classification objectives and vice-versa.
#'
#'   Note that not all possible `objective` values supported by the core XGBoost library are allowed
#'   here - for example, objectives which are a variation of another but with a different default
#'   prediction type (e.g. `multi:softmax` vs. `multi:softprob`) are not allowed, and neither are
#'   ranking objectives, nor custom objectives at the moment.
#' @param nrounds Number of boosting iterations / rounds.
#'
#'   Note that the number of default boosting rounds here is not automatically tuned, and different
#'   problems will have vastly different optimal numbers of boosting rounds.
#' @param weights Sample weights for each row in `x` and `y`. If `NULL` (the default), each row
#'   will have the same weight.
#'
#'   If not `NULL`, should be passed as a numeric vector with length matching to the number of rows in `x`.
#' @param verbosity Verbosity of printing messages. Valid values of 0 (silent), 1 (warning),
#'   2 (info), and 3 (debug).
#' @param nthreads Number of parallel threads to use. If passing zero, will use all CPU threads.
#' @param seed Seed to use for random number generation. If passing `NULL`, will draw a random
#'   number using R's PRNG system to use as seed.
#' @param monotone_constraints Optional monotonicity constraints for features.
#'
#'   Can be passed either as a named list (when `x` has column names), or as a vector. If passed
#'   as a vector and `x` has column names, will try to match the elements by name.
#'
#'   A value of `+1` for a given feature makes the model predictions / scores constrained to be
#'   a monotonically increasing function of that feature (that is, as the value of the feature
#'   increases, the model prediction cannot decrease), while a value of `-1` makes it a monotonically
#'   decreasing function. A value of zero imposes no constraint.
#'
#'   The input for `monotone_constraints` can be a subset of the columns of `x` if named, in which
#'   case the columns that are not referred to in `monotone_constraints` will be assumed to have
#'   a value of zero (no constraint imposed on the model for those features).
#'
#'   See the tutorial [Monotonic Constraints](https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html)
#'   for a more detailed explanation.
#' @param interaction_constraints Constraints for interaction representing permitted interactions.
#'   The constraints must be specified in the form of a list of vectors referencing columns in the
#'   data, e.g. `list(c(1, 2), c(3, 4, 5))` (with these numbers being column indices, numeration
#'   starting at 1 - i.e. the first sublist references the first and second columns) or
#'   `list(c("Sepal.Length", "Sepal.Width"), c("Petal.Length", "Petal.Width"))` (references
#'   columns by names), where each vector is a group of indices of features that are allowed to
#'   interact with each other.
#'
#'   See the tutorial [Feature Interaction Constraints](https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html)
#'   for more information.
#' @param feature_weights Feature weights for column sampling.
#'
#'   Can be passed either as a vector with length matching to columns of `x`, or as a named
#'   list (only if `x` has column names) with names matching to columns of 'x'. If it is a
#'   named vector, will try to match the entries to column names of `x` by name.
#'
#'   If `NULL` (the default), all columns will have the same weight.
#' @param base_margin Base margin used for boosting from existing model.
#'
#'   If passing it, will start the gradient boosting procedure from the scores that are provided
#'   here - for example, one can pass the raw scores from a previous model, or some per-observation
#'   offset, or similar.
#'
#'   Should be either a numeric vector or numeric matrix (for multi-class and multi-target objectives)
#'   with the same number of rows as `x` and number of columns corresponding to number of optimization
#'   targets, and should be in the untransformed scale (for example, for objective `binary:logistic`,
#'   it should have log-odds, not probabilities; and for objective `multi:softprob`, should have
#'   number of columns matching to number of classes in the data).
#'
#'   Note that, if it contains more than one column, then columns will not be matched by name to
#'   the corresponding `y` - `base_margin` should have the same column order that the model will use
#'   (for example, for objective `multi:softprob`, columns of `base_margin` will be matched against
#'   `levels(y)` by their position, regardless of what `colnames(base_margin)` returns).
#'
#'   If `NULL`, will start from zero, but note that for most objectives, an intercept is usually
#'   added (controllable through parameter `base_score` instead) when `base_margin` is not passed.
#' @param ... Other training parameters. See the online documentation
#'   [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) for
#'   details about possible values and what they do.
#'
#'   Note that not all possible values from the core XGBoost library are allowed as `params` for
#'   'xgboost()' - in particular, values which require an already-fitted booster object (such as
#'   `process_type`) are not accepted here.
#' @return A model object, inheriting from both `xgboost` and `xgb.Booster`. Compared to the regular
#'   `xgb.Booster` model class produced by [xgb.train()], this `xgboost` class will have an
#'
#'   additional attribute `metadata` containing information which is used for formatting prediction
#'   outputs, such as class names for classification problems.
#'
#' @examples
#' data(mtcars)
#'
#' # Fit a small regression model on the mtcars data
#' model_regression <- xgboost(mtcars[, -1], mtcars$mpg, nthreads = 1, nrounds = 3)
#' predict(model_regression, mtcars, validate_features = TRUE)
#'
#' # Task objective is determined automatically according to the type of 'y'
#' data(iris)
#' model_classif <- xgboost(iris[, -5], iris$Species, nthreads = 1, nrounds = 5)
#' predict(model_classif, iris, validate_features = TRUE)
xgboost <- function(
  x,
  y,
  objective = NULL,
  nrounds = 100L,
  weights = NULL,
  verbosity = 0L,
  nthreads = parallel::detectCores(),
  seed = 0L,
  monotone_constraints = NULL,
  interaction_constraints = NULL,
  feature_weights = NULL,
  base_margin = NULL,
  ...
) {
  # Note: '...' is a workaround, to be removed later by making all parameters be arguments
  params <- list(...)
  params <- prescreen.parameters(params)
  prescreen.objective(objective)
  use_qdm <- check.can.use.qdm(x, params)
  lst_args <- process.y.margin.and.objective(y, base_margin, objective, params)
  lst_args <- process.row.weights(weights, lst_args)
  lst_args <- process.x.and.col.args(
    x,
    monotone_constraints,
    interaction_constraints,
    feature_weights,
    lst_args,
    use_qdm
  )

  if (use_qdm && "max_bin" %in% names(params)) {
    lst_args$dmatrix_args$max_bin <- params$max_bin
  }

  nthreads <- check.nthreads(nthreads)
  lst_args$dmatrix_args$nthread <- nthreads
  lst_args$params$nthread <- nthreads
  lst_args$params$seed <- seed

  params <- c(lst_args$params, params)

  fn_dm <- if (use_qdm) xgb.QuantileDMatrix else xgb.DMatrix
  dm <- do.call(fn_dm, lst_args$dmatrix_args)
  model <- xgb.train(
    params = params,
    data = dm,
    nrounds = nrounds,
    verbose = verbosity
  )
  attributes(model)$metadata <- lst_args$metadata
  attributes(model)$call <- match.call()
  class(model) <- c("xgboost", class(model))
  return(model)
}

#' @method print xgboost
#' @export
print.xgboost <- function(x, ...) {
  cat("XGBoost model object\n")
  cat("Call:\n  ")
  print(attributes(x)$call)
  cat("Objective: ", attributes(x)$params$objective, "\n", sep = "")
  cat("Number of iterations: ", xgb.get.num.boosted.rounds(x), "\n", sep = "")
  cat("Number of features: ", xgb.num_feature(x), "\n", sep = "")

  printable_head <- function(v) {
    v_sub <- utils::head(v, 5L)
    return(
      sprintf(
        "%s%s",
        paste(v_sub, collapse = ", "),
        ifelse(length(v_sub) < length(v), ", ...", "")
      )
    )
  }

  if (NROW(attributes(x)$metadata$y_levels)) {
    cat(
      "Classes: ",
      printable_head(attributes(x)$metadata$y_levels),
      "\n",
      sep = ""
    )
  } else if (NROW(attributes(x)$params$quantile_alpha)) {
    cat(
      "Prediction quantile",
      ifelse(length(attributes(x)$params$quantile_alpha) > 1L, "s", ""),
      ": ",
      printable_head(attributes(x)$params$quantile_alpha),
      "\n",
      sep = ""
    )
  } else if (NROW(attributes(x)$metadata$y_names)) {
    cat(
      "Prediction targets: ",
      printable_head(attributes(x)$metadata$y_names),
      "\n",
      sep = ""
    )
  } else if (attributes(x)$metadata$n_targets > 1L) {
    cat(
      "Number of predition targets: ",
      attributes(x)$metadata$n_targets,
      "\n",
      sep = ""
    )
  }

  return(x)
}


#' Training part from Mushroom Data Set
#'
#' This data set is originally from the Mushroom data set,
#' UCI Machine Learning Repository.
#'
#' It includes the following fields:
#'  - `label`: The label for each record.
#'  - `data`: A sparse Matrix of 'dgCMatrix' class with 126 columns.
#'
#' @references
#' <https://archive.ics.uci.edu/ml/datasets/Mushroom>
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' <http://archive.ics.uci.edu/ml>. Irvine, CA: University of California,
#' School of Information and Computer Science.
#'
#' @docType data
#' @keywords datasets
#' @name agaricus.train
#' @usage data(agaricus.train)
#' @format A list containing a label vector, and a dgCMatrix object with 6513
#' rows and 127 variables
NULL

#' Test part from Mushroom Data Set
#'
#' This data set is originally from the Mushroom data set,
#' UCI Machine Learning Repository.
#'
#' It includes the following fields:
#'  - `label`: The label for each record.
#'  - `data`: A sparse Matrix of 'dgCMatrix' class with 126 columns.
#'
#' @references
#' <https://archive.ics.uci.edu/ml/datasets/Mushroom>
#'
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
#' <http://archive.ics.uci.edu/ml>. Irvine, CA: University of California,
#' School of Information and Computer Science.
#'
#' @docType data
#' @keywords datasets
#' @name agaricus.test
#' @usage data(agaricus.test)
#' @format A list containing a label vector, and a dgCMatrix object with 1611
#' rows and 126 variables
NULL

# Various imports
#' @importClassesFrom Matrix dgCMatrix dgRMatrix CsparseMatrix
#' @importFrom Matrix sparse.model.matrix
#' @importFrom data.table data.table
#' @importFrom data.table is.data.table
#' @importFrom data.table as.data.table
#' @importFrom data.table :=
#' @importFrom data.table rbindlist
#' @importFrom data.table setkey
#' @importFrom data.table setkeyv
#' @importFrom data.table setnames
#' @importFrom jsonlite fromJSON
#' @importFrom jsonlite toJSON
#' @importFrom methods new
#' @importFrom utils object.size str tail
#' @importFrom stats coef
#' @importFrom stats predict
#' @importFrom stats median
#' @importFrom stats sd
#' @importFrom stats variable.names
#' @importFrom utils head
#' @importFrom graphics barplot
#' @importFrom graphics lines
#' @importFrom graphics points
#' @importFrom graphics grid
#' @importFrom graphics par
#' @importFrom graphics title
#' @importFrom grDevices rgb
#'
#' @import methods
#' @useDynLib xgboost, .registration = TRUE
NULL
