prescreen.objective <- function(objective) {
  if (!is.null(objective)) {
    if (!is.character(objective) || length(objective) != 1L || is.na(objective)) {
      stop("'objective' must be a single character/string variable.")
    }

    if (objective %in% .OBJECTIVES_NON_DEFAULT_MODE()) {
      stop(
        "Objectives with non-default prediction mode (",
        paste(.OBJECTIVES_NON_DEFAULT_MODE(), collapse = ", "),
        ") are not supported in 'xgboost()'. Try 'xgb.train()'."
      )
    }

    if (objective %in% .RANKING_OBJECTIVES()) {
      stop("Ranking objectives are not supported in 'xgboost()'. Try 'xgb.train()'.")
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

check.can.use.qdm <- function(x, params, eval_set) {
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
  if (NROW(eval_set)) {
    return(FALSE)
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

process.eval.set <- function(eval_set, lst_args) {
  if (!NROW(eval_set)) {
    return(NULL)
  }
  nrows <- nrow(lst_args$dmatrix_args$data)
  is_classif <- hasName(lst_args$metadata, "y_levels")
  processed_y <- lst_args$dmatrix_args$label
  eval_set <- as.vector(eval_set)
  if (length(eval_set) == 1L) {

    eval_set <- as.numeric(eval_set)
    if (is.na(eval_set) || eval_set < 0 || eval_set >= 1) {
      stop("'eval_set' as a fraction must be a number between zero and one (non-inclusive).")
    }
    if (eval_set == 0) {
      return(NULL)
    }
    nrow_eval <- as.integer(round(nrows * eval_set, 0))
    if (nrow_eval < 1) {
      warning(
        "Desired 'eval_set' fraction amounts to zero observations.",
        " Will not create evaluation set."
      )
      return(NULL)
    }
    nrow_train <- nrows - nrow_eval
    if (nrow_train < 2L) {
      stop("Desired 'eval_set' fraction would leave less than 2 observations for training data.")
    }
    if (is_classif && nrow_train < length(lst_args$metadata$y_levels)) {
      stop("Desired 'eval_set' fraction would not leave enough samples for each class of 'y'.")
    }

    seed <- lst_args$params$seed
    if (!is.null(seed)) {
      set.seed(seed)
    }

    idx_shuffled <- sample(nrows, nrows, replace = FALSE)
    idx_eval <- idx_shuffled[seq(1L, nrow_eval)]
    idx_train <- idx_shuffled[seq(nrow_eval + 1L, nrows)]
    # Here we want the training set to include all of the classes of 'y' for classification
    # objectives. If that condition doesn't hold with the random sample, then it forcibly
    # makes a new random selection in such a way that the condition would always hold, by
    # first sampling one random example of 'y' for training and then choosing the evaluation
    # set from the remaining rows. The procedure here is quite inefficient, but there aren't
    # enough random-related functions in base R to be able to construct an efficient version.
    if (is_classif && length(unique(processed_y[idx_train])) < length(lst_args$metadata$y_levels)) {
      # These are defined in order to avoid NOTEs from CRAN checks
      # when using non-standard data.table evaluation with column names.
      idx <- NULL
      y <- NULL
      ranked_idx <- NULL
      chosen <- NULL

      dt <- data.table::data.table(y = processed_y, idx = seq(1L, nrows))[
        , .(
            ranked_idx = seq(1L, .N),
            chosen = rep(sample(.N, 1L), .N),
            idx
          )
        , by = y
      ]
      min_idx_train <- dt[ranked_idx == chosen, idx]
      rem_idx <- dt[ranked_idx != chosen, idx]
      if (length(rem_idx) == nrow_eval) {
        idx_train <- min_idx_train
        idx_eval <- rem_idx
      } else {
        rem_idx <- rem_idx[sample(length(rem_idx), length(rem_idx), replace = FALSE)]
        idx_eval <- rem_idx[seq(1L, nrow_eval)]
        idx_train <- c(min_idx_train, rem_idx[seq(nrow_eval + 1L, length(rem_idx))])
      }
    }

  } else {

    if (any(eval_set != floor(eval_set))) {
      stop("'eval_set' as indices must contain only integers.")
    }
    eval_set <- as.integer(eval_set)
    idx_min <- min(eval_set)
    if (is.na(idx_min) || idx_min < 1L) {
      stop("'eval_set' contains invalid indices.")
    }
    idx_max <- max(eval_set)
    if (is.na(idx_max) || idx_max > nrows) {
      stop("'eval_set' contains row indices beyond the size of the input data.")
    }
    idx_train <- seq(1L, nrows)[-eval_set]
    if (is_classif && length(unique(processed_y[idx_train])) < length(lst_args$metadata$y_levels)) {
      warning("'eval_set' indices will leave some classes of 'y' outside of the training data.")
    }
    idx_eval <- eval_set

  }

  # Note: slicing is done in the constructed DMatrix object instead of in the
  # original input, because objects from 'Matrix' might change class after
  # being sliced (e.g. 'dgRMatrix' turns into 'dgCMatrix').
  return(list(idx_train = idx_train, idx_eval = idx_eval))
}

check.early.stopping.rounds <- function(early_stopping_rounds, eval_set) {
  if (is.null(early_stopping_rounds)) {
    return(NULL)
  }
  if (is.null(eval_set)) {
    stop("'early_stopping_rounds' requires passing 'eval_set'.")
  }
  if (NROW(early_stopping_rounds) != 1L) {
    stop("'early_stopping_rounds' must be NULL or an integer greater than zero.")
  }
  early_stopping_rounds <- as.integer(early_stopping_rounds)
  if (is.na(early_stopping_rounds) || early_stopping_rounds <= 0L) {
    stop(
      "'early_stopping_rounds' must be NULL or an integer greater than zero. Got: ",
      early_stopping_rounds
    )
  }
  return(early_stopping_rounds)
}

# nolint start: line_length_linter.
#' Fit XGBoost Model
#'
#' @export
#' @description
#' Fits an XGBoost model (boosted decision tree ensemble) to given x/y data.
#'
#' See the tutorial [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
#' for a longer explanation of what XGBoost does, and the rest of the
#' [XGBoost Tutorials](https://xgboost.readthedocs.io/en/latest/tutorials/index.html) for further
#' explanations XGBoost's features and usage.
#'
#' This function is intended to provide a user-friendly interface for XGBoost that follows
#' R's conventions for model fitting and predictions, but which doesn't expose all of the
#' possible functionalities of the core XGBoost library.
#'
#' See [xgb.train()] for a more flexible low-level alternative which is similar across different
#' language bindings of XGBoost and which exposes additional functionalities such as training on
#' external memory data and learning-to-rank objectives.
#'
#' By default, most of the parameters here have a value of `NULL`, which signals XGBoost to use its
#' default value. Default values are automatically determined by the XGBoost core library, and are
#' subject to change over XGBoost library versions. Some of them might differ according to the
#' booster type (e.g. defaults for regularization are different for linear and tree-based boosters).
#' See [xgb.params()] and the [online documentation](https://xgboost.readthedocs.io/en/latest/parameter.html)
#' for more details about parameters - but note that some of the parameters are not supported in
#' the `xgboost()` interface.
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
#' by name as a string / character (e.g. `reg:absoluteerror`). See the
#' [Learning Task Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters)
#' page and the [xgb.params()] documentation for more detailed information on allowed values.
#'
#' If `NULL` (the default), will be automatically determined from `y` according to the following
#' logic:
#' - If `y` is a factor with 2 levels, will use `binary:logistic`.
#' - If `y` is a factor with more than 2 levels, will use `multi:softprob` (number of classes
#'   will be determined automatically, should not be passed under `params`).
#' - If `y` is a `Surv` object from the `survival` package, will use `survival:aft` (note that
#'   the only types supported are left / right / interval censored).
#' - Otherwise, will use `reg:squarederror`.
#'
#' If `objective` is not `NULL`, it must match with the type of `y` - e.g. `factor` types of `y`
#' can only be used with classification objectives and vice-versa.
#'
#' Note that not all possible `objective` values supported by the core XGBoost library are allowed
#' here - for example, objectives which are a variation of another but with a different default
#' prediction type (e.g. `multi:softmax` vs. `multi:softprob`) are not allowed, and neither are
#' ranking objectives, nor custom objectives at the moment.
#'
#' Supported values are:
#' - `"reg:squarederror"`: regression with squared loss.
#' - `"reg:squaredlogerror"`: regression with squared log loss \eqn{\frac{1}{2}[log(pred + 1) - log(label + 1)]^2}.  All input labels are required to be greater than -1.  Also, see metric `rmsle` for possible issue  with this objective.
#' - `"reg:pseudohubererror"`: regression with Pseudo Huber loss, a twice differentiable alternative to absolute loss.
#' - `"reg:absoluteerror"`: Regression with L1 error. When tree model is used, leaf value is refreshed after tree construction. If used in distributed training, the leaf value is calculated as the mean value from all workers, which is not guaranteed to be optimal.
#' - `"reg:quantileerror"`: Quantile loss, also known as "pinball loss". See later sections for its parameter and [Quantile Regression](https://xgboost.readthedocs.io/en/latest/python/examples/quantile_regression.html#sphx-glr-python-examples-quantile-regression-py) for a worked example.
#' - `"binary:logistic"`: logistic regression for binary classification, output probability
#' - `"binary:hinge"`: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
#' - `"count:poisson"`: Poisson regression for count data, output mean of Poisson distribution.
#'   `"max_delta_step"` is set to 0.7 by default in Poisson regression (used to safeguard optimization)
#' - `"survival:cox"`: Cox regression for right censored survival time data (negative values are considered right censored).
#'
#'   Note that predictions are returned on the hazard ratio scale (i.e., as HR = exp(marginal_prediction) in the proportional hazard function `h(t) = h0(t) * HR`).
#' - `"survival:aft"`: Accelerated failure time model for censored survival time data.
#' See [Survival Analysis with Accelerated Failure Time](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html) for details.
#' - `"multi:softprob"`: multi-class classification throgh multinomial logistic likelihood.
#' - `"reg:gamma"`: gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be [gamma-distributed](https://en.wikipedia.org/wiki/Gamma_distribution#Occurrence_and_applications).
#' - `"reg:tweedie"`: Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be [Tweedie-distributed](https://en.wikipedia.org/wiki/Tweedie_distribution#Occurrence_and_applications).
#'
#' The following values are \bold{NOT} supported by `xgboost`, but are supported by [xgb.train()]
#' (see [xgb.params()] for details):
#' - `"reg:logistic"`
#' - `"binary:logitraw"`
#' - `"multi:softmax"`
#' - `"rank:ndcg"`
#' - `"rank:map"`
#' - `"rank:pairwise"`
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
#' @param monitor_training Whether to monitor objective optimization progress on the input data.
#' Note that same 'x' and 'y' data are used for both model fitting and evaluation.
#' @param eval_set Subset of the data to use as evaluation set. Can be passed as:
#' - A vector of row indices (base-1 numeration) indicating the observations that are to be designed
#'   as evaluation data.
#' - A number between zero and one indicating a random fraction of the input data to use as
#'   evaluation data. Note that the selection will be done uniformly at random, regardless of
#'   argument `weights`.
#'
#' If passed, this subset of the data will be excluded from the training procedure, and the
#' evaluation metric(s) supplied under `eval_metric` will be calculated on this dataset after each
#' boosting iteration (pass `verbosity>0` to have these metrics printed during training). If
#' `eval_metric` is not passed, a default metric will be selected according to `objective`.
#'
#' If passing a fraction, in classification problems, the evaluation set will be chosen in such a
#' way that at least one observation of each class will be kept in the training data.
#'
#' For more elaborate evaluation variants (e.g. custom metrics, multiple evaluation sets, etc.),
#' one might want to use [xgb.train()] instead.
#' @param early_stopping_rounds Number of boosting rounds after which training will be stopped
#' if there is no improvement in performance (as measured by the last metric passed under
#' `eval_metric`, or by the default metric for the objective if `eval_metric` is not passed) on the
#' evaluation data from `eval_set`. Must pass `eval_set` in order to use this functionality.
#'
#' If `NULL`, early stopping will not be used.
#' @param print_every_n When passing `verbosity>0` and either `monitor_training=TRUE` or `eval_set`,
#' evaluation logs (metrics calculated on the training and/or evaluation data) will be printed every
#' nth iteration according to the value passed here. The first and last iteration are always
#' included regardless of this 'n'.
#'
#' Only has an effect when passing `verbosity>0`.
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
#' @param min_split_loss (for Tree Booster) (default=0, alias: `gamma`)
#' Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger `min_split_loss` is, the more conservative the algorithm will be. Note that a tree where no splits were made might still contain a single terminal node with a non-zero score.
#'
#' range: \eqn{[0, \infty)}
#' @param learning_rate (alias: `eta`)
#' Step size shrinkage used in update to prevent overfitting. After each boosting step, we can directly get the weights of new features, and `learning_rate` shrinks the feature weights to make the boosting process more conservative.
#' - range: \eqn{[0,1]}
#' - default value: 0.3 for tree-based boosters, 0.5 for linear booster.
#' @param reg_lambda (alias: `lambda`)
#' - For tree-based boosters:
#'   - L2 regularization term on weights. Increasing this value will make model more conservative.
#'   - default: 1
#'   - range: \eqn{[0, \infty]}
#' - For linear booster:
#'   - L2 regularization term on weights. Increasing this value will make model more conservative. Normalised to number of training examples.
#'   - default: 0
#'   - range: \eqn{[0, \infty)}
#' @param reg_alpha (alias: `reg_alpha`)
#' - L1 regularization term on weights. Increasing this value will make model more conservative.
#' - For the linear booster, it's normalised to number of training examples.
#' - default: 0
#' - range: \eqn{[0, \infty)}
#' @param updater (for Linear Booster) (default= `"shotgun"`)
#' Choice of algorithm to fit linear model
#' - `"shotgun"`: Parallel coordinate descent algorithm based on shotgun algorithm. Uses 'hogwild' parallelism and therefore produces a nondeterministic solution on each run.
#' - `"coord_descent"`: Ordinary coordinate descent algorithm. Also multithreaded but still produces a deterministic solution. When the `device` parameter is set to `"cuda"` or `"gpu"`, a GPU variant would be used.
#' @inheritParams xgb.params
#' @inheritParams xgb.train
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
#' predict(model_classif, iris[1:10,])
#' predict(model_classif, iris[1:10,], type = "class")
#'
#' # Can nevertheless choose a non-default objective if needed
#' model_poisson <- xgboost(
#'   mtcars[, -1], mtcars$mpg,
#'   objective = "count:poisson",
#'   nthreads = 1,
#'   nrounds = 3
#' )
#'
#' # Can calculate evaluation metrics during boosting rounds
#' data(ToothGrowth)
#' xgboost(
#'   ToothGrowth[, c("len", "dose")],
#'   ToothGrowth$supp,
#'   eval_metric = c("auc", "logloss"),
#'   eval_set = 0.2,
#'   monitor_training = TRUE,
#'   verbosity = 1,
#'   nthreads = 1,
#'   nrounds = 3
#' )
xgboost <- function(
  x,
  y,
  objective = NULL,
  nrounds = 100L,
  max_depth = NULL,
  learning_rate = NULL,
  min_child_weight = NULL,
  min_split_loss = NULL,
  reg_lambda = NULL,
  weights = NULL,
  verbosity = if (is.null(eval_set)) 0L else 1L,
  monitor_training = verbosity > 0,
  eval_set = NULL,
  early_stopping_rounds = NULL,
  print_every_n = 1L,
  eval_metric = NULL,
  nthreads = parallel::detectCores(),
  seed = 0L,
  base_margin = NULL,
  monotone_constraints = NULL,
  interaction_constraints = NULL,
  reg_alpha = NULL,
  max_bin = NULL,
  max_leaves = NULL,
  booster = NULL,
  subsample = NULL,
  sampling_method = NULL,
  feature_weights = NULL,
  colsample_bytree = NULL,
  colsample_bylevel = NULL,
  colsample_bynode = NULL,
  tree_method = NULL,
  max_delta_step = NULL,
  scale_pos_weight = NULL,
  updater = NULL,
  grow_policy = NULL,
  num_parallel_tree = NULL,
  multi_strategy = NULL,
  base_score = NULL,
  seed_per_iteration = NULL,
  device = NULL,
  disable_default_eval_metric = NULL,
  use_rmm = NULL,
  max_cached_hist_node = NULL,
  extmem_single_page = NULL,
  max_cat_to_onehot = NULL,
  max_cat_threshold = NULL,
  sample_type = NULL,
  normalize_type = NULL,
  rate_drop = NULL,
  one_drop = NULL,
  skip_drop = NULL,
  feature_selector = NULL,
  top_k = NULL,
  tweedie_variance_power = NULL,
  huber_slope = NULL,
  quantile_alpha = NULL,
  aft_loss_distribution = NULL,
  ...
) {
# nolint end
  check.deprecation(deprecated_xgboost_params, match.call(), ...)
  params <- as.list(environment())
  params <- params[
    (names(params) %in% formalArgs(xgb.params))
    & !sapply(params, is.null)
    & !(names(params) %in% c( # these undergo additional processing here
      "objective", "base_margin", "monotone_constraints", "interaction_constraints"
    ))
  ]

  prescreen.objective(objective)
  use_qdm <- check.can.use.qdm(x, params, eval_set)
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
  eval_set <- process.eval.set(eval_set, lst_args)

  if (use_qdm && hasName(params, "max_bin")) {
    lst_args$dmatrix_args$max_bin <- params$max_bin
  }

  nthreads <- check.nthreads(nthreads)
  lst_args$dmatrix_args$nthread <- nthreads
  lst_args$params$nthread <- nthreads

  params <- c(lst_args$params, params)
  params$verbosity <- verbosity

  fn_dm <- if (use_qdm) xgb.QuantileDMatrix else xgb.DMatrix
  dm <- do.call(fn_dm, lst_args$dmatrix_args)
  if (!is.null(eval_set)) {
    dm_eval <- xgb.slice.DMatrix(dm, eval_set$idx_eval)
    dm <- xgb.slice.DMatrix(dm, eval_set$idx_train)
  }
  evals <- list()
  if (monitor_training) {
    evals <- list(train = dm)
  }
  if (!is.null(eval_set)) {
    evals <- c(evals, list(eval = dm_eval))
  }
  model <- xgb.train(
    params = params,
    data = dm,
    nrounds = nrounds,
    verbose = verbosity,
    print_every_n = print_every_n,
    evals = evals
  )
  attributes(model)$metadata <- lst_args$metadata
  attributes(model)$call <- match.call()
  class(model) <- c("xgboost", class(model))
  return(model)
}

#' @title Compute predictions from XGBoost model on new data
#' @description Predict values on data based on XGBoost model.
#' @param object An XGBoost model object of class `xgboost`, as produced by function [xgboost()].
#'
#' Note that there is also a lower-level [predict.xgb.Booster()] method for models of class
#' `xgb.Booster` as produced by [xgb.train()], which can also be used for `xgboost` class models as
#' an alternative that performs fewer validations and post-processings.
#' @param newdata Data on which to compute predictions from the model passed in `object`. Supported
#' input classes are:
#' - Data Frames (class `data.frame` from base R and subclasses like `data.table`).
#' - Matrices (class `matrix` from base R).
#' - Sparse matrices from package `Matrix`, either as class `dgRMatrix` (CSR) or `dgCMatrix` (CSC).
#' - Sparse vectors from package `Matrix`, which will be interpreted as containing a single
#'   observation.
#'
#' In the case of data frames, if there are any categorical features, they should be of class
#' `factor` and should have the same levels as the `factor` columns of the data from which the model
#' was constructed.
#'
#' If there are named columns and the model was fitted to data with named columns, they will be
#' matched by name by default (see `validate_features`).
#' @param type Type of prediction to make. Supported options are:
#' - `"response"`: will output model predictions on the scale of the response variable (e.g.
#'  probabilities of belonging to the last class in the case of binary classification). Result will
#'  be either a numeric vector with length matching to rows in `newdata`, or a numeric matrix with
#'  shape `[nrows(newdata), nscores]` (for objectives that produce more than one score per
#'  observation such as multi-class classification or multi-quantile regression).
#' - `"raw"`: will output the unprocessed boosting scores (e.g. log-odds in the case of objective
#'   `binary:logistic`). Same output shape and type as for `"response"`.
#' - `"class"`: will output the class with the highest predicted probability, returned as a `factor`
#'   (only applicable to classification objectives) with length matching to rows in `newdata`.
#' - `"leaf"`: will output the terminal node indices of each observation across each tree, as an
#'   integer matrix of shape `[nrows(newdata), ntrees]`, or as an integer array with an extra one or
#'   two dimensions, up to `[nrows(newdata), ntrees, nscores, n_parallel_trees]` for models that
#'   produce more than one score per tree and/or which have more than one parallel tree (e.g.
#'   random forests).
#'
#'   Only applicable to tree-based boosters (not `gblinear`).
#' - `"contrib"`: will produce per-feature contribution estimates towards the model score for a
#'   given observation, based on SHAP values. The contribution values are on the scale of
#'   untransformed margin (e.g., for binary classification, the values are log-odds deviations from
#'   the baseline).
#'
#'   Output will be a numeric matrix with shape `[nrows, nfeatures+1]`, with the intercept being the
#'   last feature, or a numeric array with shape `[nrows, nscores, nfeatures+1]` if the model
#'   produces more than one score per observation.
#' - `"interaction"`: similar to `"contrib"`, but computing SHAP values of contributions of
#'   interaction of each pair of features. Note that this operation might be rather expensive in
#'   terms of compute and memory.
#'
#'   Since it quadratically depends on the number of features, it is recommended to perform
#'   selection of the most important features first.
#'
#'   Output will be a numeric array of shape `[nrows, nfeatures+1, nfeatures+1]`, or shape
#'   `[nrows, nscores, nfeatures+1, nfeatures+1]` (for objectives that produce more than one score
#'   per observation).
#' @param base_margin Base margin used for boosting from existing model (raw score that gets added to
#' all observations independently of the trees in the model).
#'
#' If supplied, should be either a vector with length equal to the number of rows in `newdata`
#' (for objectives which produces a single score per observation), or a matrix with number of
#' rows matching to the number rows in `newdata` and number of columns matching to the number
#' of scores estimated by the model (e.g. number of classes for multi-class classification).
#' @param iteration_range Sequence of rounds/iterations from the model to use for prediction, specified by passing
#' a two-dimensional vector with the start and end numbers in the sequence (same format as R's `seq` - i.e.
#' base-1 indexing, and inclusive of both ends).
#'
#' For example, passing `c(1,20)` will predict using the first twenty iterations, while passing `c(1,1)` will
#' predict using only the first one.
#'
#' If passing `NULL`, will either stop at the best iteration if the model used early stopping, or use all
#' of the iterations (rounds) otherwise.
#'
#' If passing "all", will use all of the rounds regardless of whether the model had early stopping or not.
#'
#' Not applicable to `gblinear` booster.
#' @param validate_features Validate that the feature names in the data match to the feature names
#' in the column, and reorder them in the data otherwise.
#'
#' If passing `FALSE`, it is assumed that the feature names and types are the same,
#' and come in the same order as in the training data.
#'
#' Be aware that this only applies to column names and not to factor levels in categorical columns.
#'
#' Note that this check might add some sizable latency to the predictions, so it's
#' recommended to disable it for performance-sensitive applications.
#' @param ... Not used.
#' @return Either a numeric vector (for 1D outputs), numeric matrix (for 2D outputs), numeric array
#' (for 3D and higher), or `factor` (for class predictions). See documentation for parameter `type`
#' for details about what the output type and shape will be.
#' @method predict xgboost
#' @export
#' @examples
#' data("ToothGrowth")
#' y <- ToothGrowth$supp
#' x <- ToothGrowth[, -2L]
#' model <- xgboost(x, y, nthreads = 1L, nrounds = 3L, max_depth = 2L)
#' pred_prob <- predict(model, x[1:5, ], type = "response")
#' pred_raw <- predict(model, x[1:5, ], type = "raw")
#' pred_class <- predict(model, x[1:5, ], type = "class")
#'
#' # Relationships between these
#' manual_probs <- 1 / (1 + exp(-pred_raw))
#' manual_class <- ifelse(manual_probs < 0.5, levels(y)[1], levels(y)[2])
#'
#' # They should match up to numerical precision
#' round(pred_prob, 6) == round(manual_probs, 6)
#' pred_class == manual_class
predict.xgboost <- function(
  object,
  newdata,
  type = "response",
  base_margin = NULL,
  iteration_range = NULL,
  validate_features = TRUE,
  ...
) {
  if (inherits(newdata, "xgb.DMatrix")) {
    stop(
      "Predictions on 'xgb.DMatrix' objects are not supported with 'xgboost' class.",
      " Try 'xgb.train' or 'predict.xgb.Booster'."
    )
  }

  outputmargin <- FALSE
  predleaf <- FALSE
  predcontrib <- FALSE
  predinteraction <- FALSE
  pred_class <- FALSE
  strict_shape <- FALSE
  allowed_types <- c(
    "response",
    "raw",
    "class",
    "leaf",
    "contrib",
    "interaction"
  )
  type <- head(type, 1L)
  if (!is.character(type) || !(type %in% allowed_types)) {
    stop("'type' must be one of: ", paste(allowed_types, collapse = ", "))
  }

  if (type != "response")  {
    switch(
      type,
      "raw" = {
        outputmargin <- TRUE
      }, "class" = {
        if (is.null(attributes(object)$metadata$y_levels)) {
          stop("Prediction type 'class' is only for classification objectives.")
        }
        pred_class <- TRUE
        outputmargin <- TRUE
      }, "leaf" = {
        predleaf <- TRUE
        strict_shape <- TRUE # required for 3D and 4D outputs
      }, "contrib" = {
        predcontrib <- TRUE
      }, "interaction" = {
        predinteraction <- TRUE
      }
    )
  }
  out <- predict.xgb.Booster(
    object,
    newdata,
    outputmargin = outputmargin,
    predleaf = predleaf,
    predcontrib = predcontrib,
    predinteraction = predinteraction,
    iterationrange = iteration_range,
    strict_shape = strict_shape,
    validate_features = validate_features,
    base_margin = base_margin
  )

  if (strict_shape) {
    # Should only end up here for leaf predictions
    out_dims <- dim(out)
    dims_remove <- integer()
    if (out_dims[3L] == 1L) {
      dims_remove <- c(dims_remove, -3L)
    }
    if (length(out_dims) >= 4L && out_dims[4L] == 1L) {
      dims_remove <- c(dims_remove, -4L)
    }
    if (length(dims_remove)) {
      new_dimnames <- dimnames(out)[dims_remove]
      dim(out) <- out_dims[dims_remove]
      dimnames(out) <- new_dimnames
    }
  }

  if (pred_class) {

    if (is.null(dim(out))) {
      out <- as.integer(out >= 0) + 1L
    } else {
      out <- max.col(out, ties.method = "first")
    }
    attr_out <- attributes(out)
    attr_out$class <- "factor"
    attr_out$levels <- attributes(object)$metadata$y_levels
    attributes(out) <- attr_out

  } else if (NCOL(out) > 1L || (strict_shape && length(dim(out)) >= 3L)) {

    names_use <- NULL
    if (NROW(attributes(object)$metadata$y_levels) > 2L) {
      names_use <- attributes(object)$metadata$y_levels
    } else if (NROW(attributes(object)$metadata$y_names)) {
      names_use <- attributes(object)$metadata$y_names
    } else if (NROW(attributes(object)$params$quantile_alpha) > 1L) {
      names_use <- paste0("q", attributes(object)$params$quantile_alpha)
      if (anyDuplicated(names_use)) {
        warning("Cannot add quantile names to output due to clashes in their character conversions")
        names_use <- NULL
      }
    }
    if (NROW(names_use)) {
      dimnames_out <- dimnames(out)
      dim_with_names <- if (type == "leaf") 3L else 2L
      dimnames_out[[dim_with_names]] <- names_use
      .Call(XGSetArrayDimNamesInplace_R, out, dimnames_out)
    }

  }

  return(out)
}

#' @title Print info from XGBoost model
#' @description Prints basic properties of an XGBoost model object.
#' @param x An XGBoost model object of class `xgboost`, as produced by function [xgboost()].
#' @param ... Not used.
#' @return Same object `x`, after printing its info.
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
