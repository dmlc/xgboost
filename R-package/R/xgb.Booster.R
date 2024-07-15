# Construct an internal xgboost Booster and get its current number of rounds.
# internal utility function
# Note: the number of rounds in the C booster gets reset to zero when changing
# key booster parameters like 'process_type=update', but in some cases, when
# replacing previous iterations, it needs to make a check that the new number
# of iterations doesn't exceed the previous ones, hence it keeps track of the
# current number of iterations before resetting the parameters in order to
# perform the check later on.
xgb.Booster <- function(params, cachelist, modelfile) {
  if (typeof(cachelist) != "list" ||
      !all(vapply(cachelist, inherits, logical(1), what = 'xgb.DMatrix'))) {
    stop("cachelist must be a list of xgb.DMatrix objects")
  }
  ## Load existing model, dispatch for on disk model file and in memory buffer
  if (!is.null(modelfile)) {
    if (is.character(modelfile)) {
      ## A filename
      bst <- .Call(XGBoosterCreate_R, cachelist)
      modelfile <- path.expand(modelfile)
      .Call(XGBoosterLoadModel_R, xgb.get.handle(bst), enc2utf8(modelfile[1]))
      niter <- xgb.get.num.boosted.rounds(bst)
      if (length(params) > 0) {
        xgb.parameters(bst) <- params
      }
      return(list(bst = bst, niter = niter))
    } else if (is.raw(modelfile)) {
      ## A memory buffer
      bst <- xgb.load.raw(modelfile)
      niter <- xgb.get.num.boosted.rounds(bst)
      xgb.parameters(bst) <- params
      return(list(bst = bst, niter = niter))
    } else if (inherits(modelfile, "xgb.Booster")) {
      ## A booster object
      bst <- .Call(XGDuplicate_R, modelfile)
      niter <- xgb.get.num.boosted.rounds(bst)
      xgb.parameters(bst) <- params
      return(list(bst = bst, niter = niter))
    } else {
      stop("modelfile must be either character filename, or raw booster dump, or xgb.Booster object")
    }
  }
  ## Create new model
  bst <- .Call(XGBoosterCreate_R, cachelist)
  if (length(params) > 0) {
    xgb.parameters(bst) <- params
  }
  return(list(bst = bst, niter = 0L))
}

# Check whether xgb.Booster handle is null
# internal utility function
is.null.handle <- function(handle) {
  if (is.null(handle)) return(TRUE)

  if (!inherits(handle, "externalptr"))
    stop("argument type must be 'externalptr'")

  return(.Call(XGCheckNullPtr_R, handle))
}

# Return a verified to be valid handle out of xgb.Booster
# internal utility function
xgb.get.handle <- function(object) {
  if (inherits(object, "xgb.Booster")) {
    handle <- object$ptr
    if (is.null(handle) || !inherits(handle, "externalptr")) {
      stop("'xgb.Booster' object is corrupted or is from an incompatible xgboost version.")
    }
  } else {
    stop("argument must be an 'xgb.Booster' object.")
  }
  if (is.null.handle(handle)) {
    stop("invalid 'xgb.Booster' (blank 'externalptr').")
  }
  return(handle)
}

#' Predict method for XGBoost model
#'
#' Predict values on data based on xgboost model.
#'
#' @param object Object of class `xgb.Booster`.
#' @param newdata Takes `data.frame`, `matrix`, `dgCMatrix`, `dgRMatrix`, `dsparseVector`,
#'        local data file, or `xgb.DMatrix`.
#'
#'        For single-row predictions on sparse data, it's recommended to use CSR format. If passing
#'        a sparse vector, it will take it as a row vector.
#'
#'        Note that, for repeated predictions on the same data, one might want to create a DMatrix to
#'        pass here instead of passing R types like matrices or data frames, as predictions will be
#'        faster on DMatrix.
#'
#'        If `newdata` is a `data.frame`, be aware that:\itemize{
#'        \item Columns will be converted to numeric if they aren't already, which could potentially make
#'              the operation slower than in an equivalent `matrix` object.
#'        \item The order of the columns must match with that of the data from which the model was fitted
#'              (i.e. columns will not be referenced by their names, just by their order in the data).
#'        \item If the model was fitted to data with categorical columns, these columns must be of
#'              `factor` type here, and must use the same encoding (i.e. have the same levels).
#'        \item If `newdata` contains any `factor` columns, they will be converted to base-0
#'              encoding (same as during DMatrix creation) - hence, one should not pass a `factor`
#'              under a column which during training had a different type.
#'        }
#' @param missing Float value that represents missing values in data (e.g., 0 or some other extreme value).
#'
#'        This parameter is not used when `newdata` is an `xgb.DMatrix` - in such cases, should pass
#'        this as an argument to the DMatrix constructor instead.
#' @param outputmargin Whether the prediction should be returned in the form of original untransformed
#'        sum of predictions from boosting iterations' results. E.g., setting `outputmargin=TRUE` for
#'        logistic regression would return log-odds instead of probabilities.
#' @param predleaf Whether to predict per-tree leaf indices.
#' @param predcontrib Whether to return feature contributions to individual predictions (see Details).
#' @param approxcontrib Whether to use a fast approximation for feature contributions (see Details).
#' @param predinteraction Whether to return contributions of feature interactions to individual predictions (see Details).
#' @param reshape Whether to reshape the vector of predictions to matrix form when there are several
#'        prediction outputs per case. No effect if `predleaf`, `predcontrib`,
#'        or `predinteraction` is `TRUE`.
#' @param training Whether the prediction result is used for training. For dart booster,
#'        training predicting will perform dropout.
#' @param iterationrange Sequence of rounds/iterations from the model to use for prediction, specified by passing
#'        a two-dimensional vector with the start and end numbers in the sequence (same format as R's `seq` - i.e.
#'        base-1 indexing, and inclusive of both ends).
#'
#'        For example, passing `c(1,20)` will predict using the first twenty iterations, while passing `c(1,1)` will
#'        predict using only the first one.
#'
#'        If passing `NULL`, will either stop at the best iteration if the model used early stopping, or use all
#'        of the iterations (rounds) otherwise.
#'
#'        If passing "all", will use all of the rounds regardless of whether the model had early stopping or not.
#' @param strict_shape Default is `FALSE`. When set to `TRUE`, the output
#'        type and shape of predictions are invariant to the model type.
#' @param base_margin Base margin used for boosting from existing model.
#'
#'        Note that, if `newdata` is an `xgb.DMatrix` object, this argument will
#'        be ignored as it needs to be added to the DMatrix instead (e.g. by passing it as
#'        an argument in its constructor, or by calling \link{setinfo.xgb.DMatrix}).
#'
#' @param validate_features When `TRUE`, validate that the Booster's and newdata's feature_names
#'        match (only applicable when both `object` and `newdata` have feature names).
#'
#'        If the column names differ and `newdata` is not an `xgb.DMatrix`, will try to reorder
#'        the columns in `newdata` to match with the booster's.
#'
#'        If the booster has feature types and `newdata` is either an `xgb.DMatrix` or `data.frame`,
#'        will additionally verify that categorical columns are of the correct type in `newdata`,
#'        throwing an error if they do not match.
#'
#'        If passing `FALSE`, it is assumed that the feature names and types are the same,
#'        and come in the same order as in the training data.
#'
#'        Note that this check might add some sizable latency to the predictions, so it's
#'        recommended to disable it for performance-sensitive applications.
#' @param ... Not used.
#'
#' @details
#'
#' Note that `iterationrange` would currently do nothing for predictions from "gblinear",
#' since "gblinear" doesn't keep its boosting history.
#'
#' One possible practical applications of the `predleaf` option is to use the model
#' as a generator of new features which capture non-linearity and interactions,
#' e.g., as implemented in [xgb.create.features()].
#'
#' Setting `predcontrib = TRUE` allows to calculate contributions of each feature to
#' individual predictions. For "gblinear" booster, feature contributions are simply linear terms
#' (feature_beta * feature_value). For "gbtree" booster, feature contributions are SHAP
#' values (Lundberg 2017) that sum to the difference between the expected output
#' of the model and the current prediction (where the hessian weights are used to compute the expectations).
#' Setting `approxcontrib = TRUE` approximates these values following the idea explained
#' in \url{http://blog.datadive.net/interpreting-random-forests/}.
#'
#' With `predinteraction = TRUE`, SHAP values of contributions of interaction of each pair of features
#' are computed. Note that this operation might be rather expensive in terms of compute and memory.
#' Since it quadratically depends on the number of features, it is recommended to perform selection
#' of the most important features first. See below about the format of the returned results.
#'
#' The `predict()` method uses as many threads as defined in `xgb.Booster` object (all by default).
#' If you want to change their number, assign a new number to `nthread` using [xgb.parameters<-()].
#' Note that converting a matrix to [xgb.DMatrix()] uses multiple threads too.
#'
#' @return
#' The return type depends on `strict_shape`. If `FALSE` (default):
#' - For regression or binary classification: A vector of length `nrows(newdata)`.
#' - For multiclass classification: A vector of length `num_class * nrows(newdata)` or
#'   a `(nrows(newdata), num_class)` matrix, depending on the `reshape` value.
#' - When `predleaf = TRUE`: A matrix with one column per tree.
#' - When `predcontrib = TRUE`: When not multiclass, a matrix with
#' ` num_features + 1` columns. The last "+ 1" column corresponds to the baseline value.
#'   In the multiclass case, a list of `num_class` such matrices.
#'   The contribution values are on the scale of untransformed margin
#'   (e.g., for binary classification, the values are log-odds deviations from the baseline).
#' - When `predinteraction = TRUE`: When not multiclass, the output is a 3d array of
#'   dimension `c(nrow, num_features + 1, num_features + 1)`. The off-diagonal (in the last two dimensions)
#'   elements represent different feature interaction contributions. The array is symmetric WRT the last
#'   two dimensions. The "+ 1" columns corresponds to the baselines. Summing this array along the last dimension should
#'   produce practically the same result as `predcontrib = TRUE`.
#'   In the multiclass case, a list of `num_class` such arrays.
#'
#' When `strict_shape = TRUE`, the output is always an array:
#' - For normal predictions, the output has dimension `(num_class, nrow(newdata))`.
#' - For `predcontrib = TRUE`, the dimension is `(ncol(newdata) + 1, num_class, nrow(newdata))`.
#' - For `predinteraction = TRUE`, the dimension is `(ncol(newdata) + 1, ncol(newdata) + 1, num_class, nrow(newdata))`.
#' - For `predleaf = TRUE`, the dimension is `(n_trees_in_forest, num_class, n_iterations, nrow(newdata))`.
#' @seealso [xgb.train()]
#' @references
#' 1. Scott M. Lundberg, Su-In Lee, "A Unified Approach to Interpreting Model Predictions",
#'   NIPS Proceedings 2017, \url{https://arxiv.org/abs/1705.07874}
#' 2. Scott M. Lundberg, Su-In Lee, "Consistent feature attribution for tree ensembles",
#'   \url{https://arxiv.org/abs/1706.06060}
#'
#' @examples
#' ## binary classification:
#'
#' data(agaricus.train, package = "xgboost")
#' data(agaricus.test, package = "xgboost")
#'
#' ## Keep the number of threads to 2 for examples
#' nthread <- 2
#' data.table::setDTthreads(nthread)
#'
#' train <- agaricus.train
#' test <- agaricus.test
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(train$data, label = train$label),
#'   max_depth = 2,
#'   eta = 0.5,
#'   nthread = nthread,
#'   nrounds = 5,
#'   objective = "binary:logistic"
#' )
#'
#' # use all trees by default
#' pred <- predict(bst, test$data)
#' # use only the 1st tree
#' pred1 <- predict(bst, test$data, iterationrange = c(1, 1))
#'
#' # Predicting tree leafs:
#' # the result is an nsamples X ntrees matrix
#' pred_leaf <- predict(bst, test$data, predleaf = TRUE)
#' str(pred_leaf)
#'
#' # Predicting feature contributions to predictions:
#' # the result is an nsamples X (nfeatures + 1) matrix
#' pred_contr <- predict(bst, test$data, predcontrib = TRUE)
#' str(pred_contr)
#' # verify that contributions' sums are equal to log-odds of predictions (up to float precision):
#' summary(rowSums(pred_contr) - qlogis(pred))
#' # for the 1st record, let's inspect its features that had non-zero contribution to prediction:
#' contr1 <- pred_contr[1,]
#' contr1 <- contr1[-length(contr1)]    # drop intercept
#' contr1 <- contr1[contr1 != 0]        # drop non-contributing features
#' contr1 <- contr1[order(abs(contr1))] # order by contribution magnitude
#' old_mar <- par("mar")
#' par(mar = old_mar + c(0,7,0,0))
#' barplot(contr1, horiz = TRUE, las = 2, xlab = "contribution to prediction in log-odds")
#' par(mar = old_mar)
#'
#'
#' ## multiclass classification in iris dataset:
#'
#' lb <- as.numeric(iris$Species) - 1
#' num_class <- 3
#'
#' set.seed(11)
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(as.matrix(iris[, -5]), label = lb),
#'   max_depth = 4,
#'   eta = 0.5,
#'   nthread = 2,
#'   nrounds = 10,
#'   subsample = 0.5,
#'   objective = "multi:softprob",
#'   num_class = num_class
#' )
#'
#' # predict for softmax returns num_class probability numbers per case:
#' pred <- predict(bst, as.matrix(iris[, -5]))
#' str(pred)
#' # reshape it to a num_class-columns matrix
#' pred <- matrix(pred, ncol = num_class, byrow = TRUE)
#' # convert the probabilities to softmax labels
#' pred_labels <- max.col(pred) - 1
#' # the following should result in the same error as seen in the last iteration
#' sum(pred_labels != lb) / length(lb)
#'
#' # compare with predictions from softmax:
#' set.seed(11)
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(as.matrix(iris[, -5]), label = lb),
#'   max_depth = 4,
#'   eta = 0.5,
#'   nthread = 2,
#'   nrounds = 10,
#'   subsample = 0.5,
#'   objective = "multi:softmax",
#'   num_class = num_class
#' )
#'
#' pred <- predict(bst, as.matrix(iris[, -5]))
#' str(pred)
#' all.equal(pred, pred_labels)
#' # prediction from using only 5 iterations should result
#' # in the same error as seen in iteration 5:
#' pred5 <- predict(bst, as.matrix(iris[, -5]), iterationrange = c(1, 5))
#' sum(pred5 != lb) / length(lb)
#'
#' @export
predict.xgb.Booster <- function(object, newdata, missing = NA, outputmargin = FALSE,
                                predleaf = FALSE, predcontrib = FALSE, approxcontrib = FALSE, predinteraction = FALSE,
                                reshape = FALSE, training = FALSE, iterationrange = NULL, strict_shape = FALSE,
                                validate_features = FALSE, base_margin = NULL, ...) {
  if (validate_features) {
    newdata <- validate.features(object, newdata)
  }
  is_dmatrix <- inherits(newdata, "xgb.DMatrix")
  if (is_dmatrix && !is.null(base_margin)) {
    stop(
      "'base_margin' is not supported when passing 'xgb.DMatrix' as input.",
      " Should be passed as argument to 'xgb.DMatrix' constructor."
    )
  }

  use_as_df <- FALSE
  use_as_dense_matrix <- FALSE
  use_as_csr_matrix <- FALSE
  n_row <- NULL
  if (!is_dmatrix) {

    inplace_predict_supported <- !predcontrib && !predinteraction && !predleaf
    if (inplace_predict_supported) {
      booster_type <- xgb.booster_type(object)
      if (booster_type == "gblinear" || (booster_type == "dart" && training)) {
        inplace_predict_supported <- FALSE
      }
    }
    if (inplace_predict_supported) {

      if (is.matrix(newdata)) {
        use_as_dense_matrix <- TRUE
      } else if (is.data.frame(newdata)) {
        # note: since here it turns it into a non-data-frame list,
        # needs to keep track of the number of rows it had for later
        n_row <- nrow(newdata)
        newdata <- lapply(
          newdata,
          function(x) {
            if (is.factor(x)) {
              return(as.numeric(x) - 1)
            } else {
              return(as.numeric(x))
            }
          }
        )
        use_as_df <- TRUE
      } else if (inherits(newdata, "dgRMatrix")) {
        use_as_csr_matrix <- TRUE
        csr_data <- list(newdata@p, newdata@j, newdata@x, ncol(newdata))
      } else if (inherits(newdata, "dsparseVector")) {
        use_as_csr_matrix <- TRUE
        n_row <- 1L
        i <- newdata@i - 1L
        if (storage.mode(i) != "integer") {
          storage.mode(i) <- "integer"
        }
        csr_data <- list(c(0L, length(i)), i, newdata@x, length(newdata))
      }

    }

  } # if (!is_dmatrix)

  if (!is_dmatrix && !use_as_dense_matrix && !use_as_csr_matrix && !use_as_df) {
    nthread <- xgb.nthread(object)
    newdata <- xgb.DMatrix(
      newdata,
      missing = missing,
      base_margin = base_margin,
      nthread = NVL(nthread, -1)
    )
    is_dmatrix <- TRUE
  }

  if (is.null(n_row)) {
    n_row <- nrow(newdata)
  }


  if (!is.null(iterationrange)) {
    if (is.character(iterationrange)) {
      stopifnot(iterationrange == "all")
      iterationrange <- c(0, 0)
    } else {
      iterationrange[1] <- iterationrange[1] - 1 # base-0 indexing
    }
  } else {
    ## no limit is supplied, use best
    best_iteration <- xgb.best_iteration(object)
    if (is.null(best_iteration)) {
      iterationrange <- c(0, 0)
    } else {
      iterationrange <- c(0, as.integer(best_iteration) + 1L)
    }
  }
  ## Handle the 0 length values.
  box <- function(val) {
    if (length(val) == 0) {
      cval <- vector(, 1)
      cval[0] <- val
      return(cval)
    }
    return(val)
  }

  ## We set strict_shape to TRUE then drop the dimensions conditionally
  args <- list(
    training = box(training),
    strict_shape = box(TRUE),
    iteration_begin = box(as.integer(iterationrange[1])),
    iteration_end = box(as.integer(iterationrange[2])),
    type = box(as.integer(0))
  )

  set_type <- function(type) {
    if (args$type != 0) {
      stop("One type of prediction at a time.")
    }
    return(box(as.integer(type)))
  }
  if (outputmargin) {
    args$type <- set_type(1)
  }
  if (predcontrib) {
    args$type <- set_type(if (approxcontrib) 3 else 2)
  }
  if (predinteraction) {
    args$type <- set_type(if (approxcontrib) 5 else 4)
  }
  if (predleaf) {
    args$type <- set_type(6)
  }

  json_conf <- jsonlite::toJSON(args, auto_unbox = TRUE)
  if (is_dmatrix) {
    predts <- .Call(
      XGBoosterPredictFromDMatrix_R, xgb.get.handle(object), newdata, json_conf
    )
  } else if (use_as_dense_matrix) {
    predts <- .Call(
      XGBoosterPredictFromDense_R, xgb.get.handle(object), newdata, missing, json_conf, base_margin
    )
  } else if (use_as_csr_matrix) {
    predts <- .Call(
      XGBoosterPredictFromCSR_R, xgb.get.handle(object), csr_data, missing, json_conf, base_margin
    )
  } else if (use_as_df) {
    predts <- .Call(
      XGBoosterPredictFromColumnar_R, xgb.get.handle(object), newdata, missing, json_conf, base_margin
    )
  }

  names(predts) <- c("shape", "results")
  shape <- predts$shape
  arr <- predts$results

  n_ret <- length(arr)
  if (n_row != shape[1]) {
    stop("Incorrect predict shape.")
  }

  .Call(XGSetArrayDimInplace_R, arr, rev(shape))

  cnames <- if (!is.null(colnames(newdata))) c(colnames(newdata), "(Intercept)") else NULL
  n_groups <- shape[2]

  ## Needed regardless of whether strict shape is being used.
  if (predcontrib) {
    .Call(XGSetArrayDimNamesInplace_R, arr, list(cnames, NULL, NULL))
  } else if (predinteraction) {
    .Call(XGSetArrayDimNamesInplace_R, arr, list(cnames, cnames, NULL, NULL))
  }
  if (strict_shape) {
    return(arr) # strict shape is calculated by libxgboost uniformly.
  }

  if (predleaf) {
    ## Predict leaf
    if (n_ret == n_row) {
      .Call(XGSetArrayDimInplace_R, arr, c(n_row, 1L))
    } else {
      arr <- matrix(arr, nrow = n_row, byrow = TRUE)
    }
  } else if (predcontrib) {
    ## Predict contribution
    arr <- aperm(a = arr, perm = c(2, 3, 1)) # [group, row, col]
    if (n_ret == n_row) {
      .Call(XGSetArrayDimInplace_R, arr, c(n_row, 1L))
      .Call(XGSetArrayDimNamesInplace_R, arr, list(NULL, cnames))
    } else if (n_groups != 1) {
      ## turns array into list of matrices
      arr <- lapply(seq_len(n_groups), function(g) arr[g, , ])
    } else {
      ## remove the first axis (group)
      newdim <- dim(arr)[2:3]
      newdn <- dimnames(arr)[2:3]
      arr <- arr[1, , ]
      .Call(XGSetArrayDimInplace_R, arr, newdim)
      .Call(XGSetArrayDimNamesInplace_R, arr, newdn)
    }
  } else if (predinteraction) {
    ## Predict interaction
    arr <- aperm(a = arr, perm = c(3, 4, 1, 2)) # [group, row, col, col]
    if (n_ret == n_row) {
      .Call(XGSetArrayDimInplace_R, arr, c(n_row, 1L))
      .Call(XGSetArrayDimNamesInplace_R, arr, list(NULL, cnames))
    } else if (n_groups != 1) {
      ## turns array into list of matrices
      arr <- lapply(seq_len(n_groups), function(g) arr[g, , , ])
    } else {
      ## remove the first axis (group)
      arr <- arr[1, , , , drop = FALSE]
      newdim <- dim(arr)[2:4]
      newdn <- dimnames(arr)[2:4]
      .Call(XGSetArrayDimInplace_R, arr, newdim)
      .Call(XGSetArrayDimNamesInplace_R, arr, newdn)
    }
  } else {
    ## Normal prediction
    if (reshape && n_groups != 1) {
      arr <- matrix(arr, ncol = n_groups, byrow = TRUE)
    } else {
      .Call(XGSetArrayDimInplace_R, arr, NULL)
    }
  }
  return(arr)
}

validate.features <- function(bst, newdata) {
  if (is.character(newdata)) {
    # this will be encountered when passing file paths
    return(newdata)
  }
  if (inherits(newdata, "sparseVector")) {
    # in this case, newdata won't have metadata
    return(newdata)
  }
  if (is.vector(newdata)) {
    newdata <- as.matrix(newdata)
  }

  booster_names <- getinfo(bst, "feature_name")
  checked_names <- FALSE
  if (NROW(booster_names)) {

    try_reorder <- FALSE
    if (inherits(newdata, "xgb.DMatrix")) {
      curr_names <- getinfo(newdata, "feature_name")
    } else {
      curr_names <- colnames(newdata)
      try_reorder <- TRUE
    }

    if (NROW(curr_names)) {
      checked_names <- TRUE

      if (length(curr_names) != length(booster_names) || any(curr_names != booster_names)) {

        if (!try_reorder) {
          stop("Feature names in 'newdata' do not match with booster's.")
        } else {
          if (inherits(newdata, "data.table")) {
            newdata <- newdata[, booster_names, with = FALSE]
          } else {
            newdata <- newdata[, booster_names, drop = FALSE]
          }
        }

      }

    } # if (NROW(curr_names)) {

  } # if (NROW(booster_names)) {

  if (inherits(newdata, c("data.frame", "xgb.DMatrix"))) {

    booster_types <- getinfo(bst, "feature_type")
    if (!NROW(booster_types)) {
      # Note: types in the booster are optional. Other interfaces
      # might not even save it as booster attributes for example,
      # even if the model uses categorical features.
      return(newdata)
    }
    if (inherits(newdata, "xgb.DMatrix")) {
      curr_types <- getinfo(newdata, "feature_type")
      if (length(curr_types) != length(booster_types) || any(curr_types != booster_types)) {
        stop("Feature types in 'newdata' do not match with booster's.")
      }
    }
    if (inherits(newdata, "data.frame")) {
      is_factor <- sapply(newdata, is.factor)
      if (any(is_factor != (booster_types == "c"))) {
        stop(
          paste0(
            "Feature types in 'newdata' do not match with booster's for same columns (by ",
            ifelse(checked_names, "name", "position"),
            ")."
          )
        )
      }
    }

  }

  return(newdata)
}


#' @title Accessors for serializable attributes of a model
#'
#' @description These methods allow to manipulate the key-value attribute strings of an xgboost model.
#'
#' @param object Object of class `xgb.Booster`. \bold{Will be modified in-place} when assigning to it.
#' @param name A non-empty character string specifying which attribute is to be accessed.
#' @param value For `xgb.attr<-`, a value of an attribute; for `xgb.attributes<-`,
#'        it is a list (or an object coercible to a list) with the names of attributes to set
#'        and the elements corresponding to attribute values.
#'        Non-character values are converted to character.
#'        When an attribute value is not a scalar, only the first index is used.
#'        Use `NULL` to remove an attribute.
#'
#' @details
#' The primary purpose of xgboost model attributes is to store some meta data about the model.
#' Note that they are a separate concept from the object attributes in R.
#' Specifically, they refer to key-value strings that can be attached to an xgboost model,
#' stored together with the model's binary representation, and accessed later
#' (from R or any other interface).
#' In contrast, any R attribute assigned to an R object of `xgb.Booster` class
#' would not be saved by [xgb.save()] because an xgboost model is an external memory object
#' and its serialization is handled externally.
#' Also, setting an attribute that has the same name as one of xgboost's parameters wouldn't
#' change the value of that parameter for a model.
#' Use [xgb.parameters<-()] to set or change model parameters.
#'
#' The `xgb.attributes<-` setter either updates the existing or adds one or several attributes,
#' but it doesn't delete the other existing attributes.
#'
#' Important: since this modifies the booster's C object, semantics for assignment here
#' will differ from R's, as any object reference to the same booster will be modified
#' too, while assignment of R attributes through `attributes(model)$<attr> <- <value>`
#' will follow the usual copy-on-write R semantics (see \link{xgb.copy.Booster} for an
#' example of these behaviors).
#'
#' @return
#' - `xgb.attr()` returns either a string value of an attribute
#'   or `NULL` if an attribute wasn't stored in a model.
#' - `xgb.attributes()` returns a list of all attributes stored in a model
#'   or `NULL` if a model has no stored attributes.
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#' train <- agaricus.train
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(train$data, label = train$label),
#'   max_depth = 2,
#'   eta = 1,
#'   nthread = 2,
#'   nrounds = 2,
#'   objective = "binary:logistic"
#' )
#'
#' xgb.attr(bst, "my_attribute") <- "my attribute value"
#' print(xgb.attr(bst, "my_attribute"))
#' xgb.attributes(bst) <- list(a = 123, b = "abc")
#'
#' fname <- file.path(tempdir(), "xgb.ubj")
#' xgb.save(bst, fname)
#' bst1 <- xgb.load(fname)
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
  out <- .Call(XGBoosterGetAttr_R, handle, as.character(name[1]))
  if (!NROW(out) || !nchar(out)) {
    return(NULL)
  }
  if (!is.null(out)) {
    if (name %in% c("best_iteration", "best_score")) {
      out <- as.numeric(out)
    }
  }
  return(out)
}

#' @rdname xgb.attr
#' @export
`xgb.attr<-` <- function(object, name, value) {
  name <- as.character(name[1])
  if (!NROW(name) || !nchar(name)) stop("invalid attribute name")
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
  .Call(XGBoosterSetAttr_R, handle, name, value)
  return(object)
}

#' @rdname xgb.attr
#' @export
xgb.attributes <- function(object) {
  handle <- xgb.get.handle(object)
  attr_names <- .Call(XGBoosterGetAttrNames_R, handle)
  if (!NROW(attr_names)) return(list())
  out <- lapply(attr_names, function(name) xgb.attr(object, name))
  names(out) <- attr_names
  return(out)
}

#' @rdname xgb.attr
#' @export
`xgb.attributes<-` <- function(object, value) {
  a <- as.list(value)
  if (is.null(names(a)) || any(nchar(names(a)) == 0)) {
    stop("attribute names cannot be empty strings")
  }
  for (i in seq_along(a)) {
    xgb.attr(object, names(a[i])) <- a[[i]]
  }
  return(object)
}

#' @title Accessors for model parameters as JSON string
#' @details Note that assignment is performed in-place on the booster C object, which unlike assignment
#' of R attributes, doesn't follow typical copy-on-write semantics for assignment - i.e. all references
#' to the same booster will also get updated.
#'
#' See \link{xgb.copy.Booster} for an example of this behavior.
#' @param object Object of class `xgb.Booster`. \bold{Will be modified in-place} when assigning to it.
#' @param value An R list.
#' @return `xgb.config` will return the parameters as an R list.
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' ## Keep the number of threads to 1 for examples
#' nthread <- 1
#' data.table::setDTthreads(nthread)
#' train <- agaricus.train
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(train$data, label = train$label),
#'   max_depth = 2,
#'   eta = 1,
#'   nthread = nthread,
#'   nrounds = 2,
#'   objective = "binary:logistic"
#' )
#'
#' config <- xgb.config(bst)
#'
#' @rdname xgb.config
#' @export
xgb.config <- function(object) {
  handle <- xgb.get.handle(object)
  return(jsonlite::fromJSON(.Call(XGBoosterSaveJsonConfig_R, handle)))
}

#' @rdname xgb.config
#' @export
`xgb.config<-` <- function(object, value) {
  handle <- xgb.get.handle(object)
  .Call(
    XGBoosterLoadJsonConfig_R,
    handle,
    jsonlite::toJSON(value, auto_unbox = TRUE, null = "null")
  )
  return(object)
}

#' @title Accessors for model parameters
#' @description Only the setter for xgboost parameters is currently implemented.
#' @details Just like \link{xgb.attr}, this function will make in-place modifications
#' on the booster object which do not follow typical R assignment semantics - that is,
#' all references to the same booster will also be updated, unlike assingment of R
#' attributes which follow copy-on-write semantics.
#'
#' See \link{xgb.copy.Booster} for an example of this behavior.
#'
#' Be aware that setting parameters of a fitted booster related to training continuation / updates
#' will reset its number of rounds indicator to zero.
#' @param object Object of class `xgb.Booster`. \bold{Will be modified in-place}.
#' @param value A list (or an object coercible to a list) with the names of parameters to set
#'        and the elements corresponding to parameter values.
#' @return The same booster `object`, which gets modified in-place.
#' @examples
#' data(agaricus.train, package = "xgboost")
#' train <- agaricus.train
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(train$data, label = train$label),
#'   max_depth = 2,
#'   eta = 1,
#'   nthread = 2,
#'   nrounds = 2,
#'   objective = "binary:logistic"
#' )
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
  names(p) <- gsub(".", "_", names(p), fixed = TRUE)
  p <- lapply(p, function(x) {
    if (is.vector(x) && length(x) == 1) {
      return(as.character(x)[1])
    } else {
      return(jsonlite::toJSON(x, auto_unbox = TRUE))
    }
  })
  handle <- xgb.get.handle(object)
  for (i in seq_along(p)) {
    .Call(XGBoosterSetParam_R, handle, names(p[i]), p[[i]])
  }
  return(object)
}

#' @rdname getinfo
#' @export
getinfo.xgb.Booster <- function(object, name) {
  name <- as.character(head(name, 1L))
  allowed_fields <- c("feature_name", "feature_type")
  if (!(name %in% allowed_fields)) {
    stop("getinfo: name must be one of the following: ", paste(allowed_fields, collapse = ", "))
  }
  handle <- xgb.get.handle(object)
  out <- .Call(
    XGBoosterGetStrFeatureInfo_R,
    handle,
    name
  )
  if (!NROW(out)) {
    return(NULL)
  }
  return(out)
}

#' @rdname getinfo
#' @export
setinfo.xgb.Booster <- function(object, name, info) {
  name <- as.character(head(name, 1L))
  allowed_fields <- c("feature_name", "feature_type")
  if (!(name %in% allowed_fields)) {
    stop("setinfo: unknown info name ", name)
  }
  info <- as.character(info)
  handle <- xgb.get.handle(object)
  .Call(
    XGBoosterSetStrFeatureInfo_R,
    handle,
    name,
    info
  )
  return(TRUE)
}

#' @title Get number of boosting in a fitted booster
#' @param model,x A fitted `xgb.Booster` model.
#' @return The number of rounds saved in the model, as an integer.
#' @details Note that setting booster parameters related to training
#' continuation / updates through \link{xgb.parameters<-} will reset the
#' number of rounds to zero.
#' @export
#' @rdname xgb.get.num.boosted.rounds
xgb.get.num.boosted.rounds <- function(model) {
  return(.Call(XGBoosterBoostedRounds_R, xgb.get.handle(model)))
}

#' @rdname xgb.get.num.boosted.rounds
#' @export
length.xgb.Booster <- function(x) {
  return(xgb.get.num.boosted.rounds(x))
}

#' @title Slice Booster by Rounds
#' @description Creates a new booster including only a selected range of rounds / iterations
#' from an existing booster, as given by the sequence `seq(start, end, step)`.
#' @details Note that any R attributes that the booster might have, will not be copied into
#' the resulting object.
#' @param model,x A fitted `xgb.Booster` object, which is to be sliced by taking only a subset
#' of its rounds / iterations.
#' @param start Start of the slice (base-1 and inclusive, like R's \link{seq}).
#' @param end End of the slice (base-1 and inclusive, like R's \link{seq}).
#'
#' Passing a value of zero here is equivalent to passing the full number of rounds in the
#' booster object.
#' @param step Step size of the slice. Passing '1' will take every round in the sequence defined by
#' `(start, end)`, while passing '2' will take every second value, and so on.
#' @return A sliced booster object containing only the requested rounds.
#' @examples
#' data(mtcars)
#' y <- mtcars$mpg
#' x <- as.matrix(mtcars[, -1])
#' dm <- xgb.DMatrix(x, label = y, nthread = 1)
#' model <- xgb.train(data = dm, params = list(nthread = 1), nrounds = 5)
#' model_slice <- xgb.slice.Booster(model, 1, 3)
#' # Prediction for first three rounds
#' predict(model, x, predleaf = TRUE)[, 1:3]
#'
#' # The new model has only those rounds, so
#' # a full prediction from it is equivalent
#' predict(model_slice, x, predleaf = TRUE)
#' @export
#' @rdname xgb.slice.Booster
xgb.slice.Booster <- function(model, start, end = xgb.get.num.boosted.rounds(model), step = 1L) {
  # This makes the slice mimic the behavior of R's 'seq',
  # which truncates on the end of the slice when the step
  # doesn't reach it.
  if (end > start && step > 1) {
    d <- (end - start + 1) / step
    if (d != floor(d)) {
      end <- start + step * ceiling(d) - 1
    }
  }
  return(
    .Call(
      XGBoosterSlice_R,
      xgb.get.handle(model),
      start - 1,
      end,
      step
    )
  )
}

#' @export
#' @rdname xgb.slice.Booster
#' @param i The indices - must be an increasing sequence as generated by e.g. `seq(...)`.
`[.xgb.Booster` <- function(x, i) {
  if (missing(i)) {
    return(xgb.slice.Booster(x, 1, 0))
  }
  if (length(i) == 1) {
    return(xgb.slice.Booster(x, i, i))
  }
  steps <- diff(i)
  if (any(steps < 0)) {
    stop("Can only slice booster with ascending sequences.")
  }
  if (length(unique(steps)) > 1) {
    stop("Can only slice booster with fixed-step sequences.")
  }
  return(xgb.slice.Booster(x, i[1L], i[length(i)], steps[1L]))
}

#' @title Get Features Names from Booster
#' @description Returns the feature / variable / column names from a fitted
#' booster object, which are set automatically during the call to \link{xgb.train}
#' from the DMatrix names, or which can be set manually through \link{setinfo}.
#'
#' If the object doesn't have feature names, will return `NULL`.
#'
#' It is equivalent to calling `getinfo(object, "feature_name")`.
#' @param object An `xgb.Booster` object.
#' @param ... Not used.
#' @export
variable.names.xgb.Booster <- function(object, ...) {
  return(getinfo(object, "feature_name"))
}

xgb.nthread <- function(bst) {
  config <- xgb.config(bst)
  out <- strtoi(config$learner$generic_param$nthread)
  return(out)
}

xgb.booster_type <- function(bst) {
  config <- xgb.config(bst)
  out <- config$learner$learner_train_param$booster
  return(out)
}

xgb.num_class <- function(bst) {
  config <- xgb.config(bst)
  out <- strtoi(config$learner$learner_model_param$num_class)
  return(out)
}

xgb.feature_names <- function(bst) {
  return(getinfo(bst, "feature_name"))
}

xgb.feature_types <- function(bst) {
  return(getinfo(bst, "feature_type"))
}

xgb.num_feature <- function(bst) {
  handle <- xgb.get.handle(bst)
  return(.Call(XGBoosterGetNumFeature_R, handle))
}

xgb.best_iteration <- function(bst) {
  out <- xgb.attr(bst, "best_iteration")
  if (!NROW(out) || !nchar(out)) {
    out <- NULL
  }
  return(out)
}

#' @title Extract coefficients from linear booster
#' @description Extracts the coefficients from a 'gblinear' booster object,
#' as produced by \code{xgb.train} when using parameter `booster="gblinear"`.
#'
#' Note: this function will error out if passing a booster model
#' which is not of "gblinear" type.
#' @param object A fitted booster of 'gblinear' type.
#' @param ... Not used.
#' @return The extracted coefficients:\itemize{
#' \item If there's only one coefficient per column in the data, will be returned as a
#' vector, potentially containing the feature names if available, with the intercept
#' as first column.
#' \item If there's more than one coefficient per column in the data (e.g. when using
#' `objective="multi:softmax"`), will be returned as a matrix with dimensions equal
#' to `[num_features, num_cols]`, with the intercepts as first row. Note that the column
#' (classes in multi-class classification) dimension will not be named.
#' }
#'
#' The intercept returned here will include the 'base_score' parameter (unlike the 'bias'
#' or the last coefficient in the model dump, which doesn't have 'base_score' added to it),
#' hence one should get the same values from calling `predict(..., outputmargin = TRUE)` and
#' from performing a matrix multiplication with `model.matrix(~., ...)`.
#'
#' Be aware that the coefficients are obtained by first converting them to strings and
#' back, so there will always be some very small lose of precision compared to the actual
#' coefficients as used by \link{predict.xgb.Booster}.
#' @examples
#' library(xgboost)
#' data(mtcars)
#' y <- mtcars[, 1]
#' x <- as.matrix(mtcars[, -1])
#' dm <- xgb.DMatrix(data = x, label = y, nthread = 1)
#' params <- list(booster = "gblinear", nthread = 1)
#' model <- xgb.train(data = dm, params = params, nrounds = 2)
#' coef(model)
#' @export
coef.xgb.Booster <- function(object, ...) {
  return(.internal.coef.xgb.Booster(object, add_names = TRUE))
}

.internal.coef.xgb.Booster <- function(object, add_names = TRUE) {
  booster_type <- xgb.booster_type(object)
  if (booster_type != "gblinear") {
    stop("Coefficients are not defined for Booster type ", booster_type)
  }
  model_json <- jsonlite::fromJSON(rawToChar(xgb.save.raw(object, raw_format = "json")))
  base_score <- model_json$learner$learner_model_param$base_score
  num_feature <- as.numeric(model_json$learner$learner_model_param$num_feature)

  weights <- model_json$learner$gradient_booster$model$weights
  n_cols <- length(weights) / (num_feature + 1)
  if (n_cols != floor(n_cols) || n_cols < 1) {
    stop("Internal error: could not determine shape of coefficients.")
  }
  sep <- num_feature * n_cols
  coefs <- weights[seq(1, sep)]
  intercepts <- weights[seq(sep + 1, length(weights))]
  intercepts <- intercepts + as.numeric(base_score)

  if (add_names) {
    feature_names <- xgb.feature_names(object)
    if (!NROW(feature_names)) {
      # This mimics the default naming in R which names columns as "V1..N"
      # when names are needed but not available
      feature_names <- paste0("V", seq(1L, num_feature))
    }
    feature_names <- c("(Intercept)", feature_names)
  }
  if (n_cols == 1L) {
    out <- c(intercepts, coefs)
    if (add_names) {
      names(out) <- feature_names
    }
  } else {
    coefs <- matrix(coefs, nrow = num_feature, byrow = TRUE)
    dim(intercepts) <- c(1L, n_cols)
    out <- rbind(intercepts, coefs)
    if (add_names) {
      row.names(out) <- feature_names
    }
    # TODO: if a class names attributes is added,
    # should use those names here.
  }
  return(out)
}

#' @title Deep-copies a Booster Object
#' @description Creates a deep copy of an 'xgb.Booster' object, such that the
#' C object pointer contained will be a different object, and hence functions
#' like \link{xgb.attr} will not affect the object from which it was copied.
#' @param model An 'xgb.Booster' object.
#' @return A deep copy of `model` - it will be identical in every way, but C-level
#' functions called on that copy will not affect the `model` variable.
#' @examples
#' library(xgboost)
#' data(mtcars)
#' y <- mtcars$mpg
#' x <- mtcars[, -1]
#' dm <- xgb.DMatrix(x, label = y, nthread = 1)
#' model <- xgb.train(
#'   data = dm,
#'   params = list(nthread = 1),
#'   nround = 3
#' )
#'
#' # Set an arbitrary attribute kept at the C level
#' xgb.attr(model, "my_attr") <- 100
#' print(xgb.attr(model, "my_attr"))
#'
#' # Just assigning to a new variable will not create
#' # a deep copy - C object pointer is shared, and in-place
#' # modifications will affect both objects
#' model_shallow_copy <- model
#' xgb.attr(model_shallow_copy, "my_attr") <- 333
#' # 'model' was also affected by this change:
#' print(xgb.attr(model, "my_attr"))
#'
#' model_deep_copy <- xgb.copy.Booster(model)
#' xgb.attr(model_deep_copy, "my_attr") <- 444
#' # 'model' was NOT affected by this change
#' # (keeps previous value that was assigned before)
#' print(xgb.attr(model, "my_attr"))
#'
#' # Verify that the new object was actually modified
#' print(xgb.attr(model_deep_copy, "my_attr"))
#' @export
xgb.copy.Booster <- function(model) {
  if (!inherits(model, "xgb.Booster")) {
    stop("'model' must be an 'xgb.Booster' object.")
  }
  return(.Call(XGDuplicate_R, model))
}

#' @title Check if two boosters share the same C object
#' @description Checks whether two booster objects refer to the same underlying C object.
#' @details As booster objects (as returned by e.g. \link{xgb.train}) contain an R 'externalptr'
#' object, they don't follow typical copy-on-write semantics of other R objects - that is, if
#' one assigns a booster to a different variable and modifies that new variable through in-place
#' methods like \link{xgb.attr<-}, the modification will be applied to both the old and the new
#' variable, unlike typical R assignments which would only modify the latter.
#'
#' This function allows checking whether two booster objects share the same 'externalptr',
#' regardless of the R attributes that they might have.
#'
#' In order to duplicate a booster in such a way that the copy wouldn't share the same
#' 'externalptr', one can use function \link{xgb.copy.Booster}.
#' @param obj1 Booster model to compare with `obj2`.
#' @param obj2 Booster model to compare with `obj1`.
#' @return Either `TRUE` or `FALSE` according to whether the two boosters share
#' the underlying C object.
#' @seealso \link{xgb.copy.Booster}
#' @examples
#' library(xgboost)
#' data(mtcars)
#' y <- mtcars$mpg
#' x <- as.matrix(mtcars[, -1])
#' model <- xgb.train(
#'   params = list(nthread = 1),
#'   data = xgb.DMatrix(x, label = y, nthread = 1),
#'   nround = 3
#' )
#'
#' model_shallow_copy <- model
#' xgb.is.same.Booster(model, model_shallow_copy) # same C object
#'
#' model_deep_copy <- xgb.copy.Booster(model)
#' xgb.is.same.Booster(model, model_deep_copy) # different C objects
#'
#' # In-place assignments modify all references,
#' # but not full/deep copies of the booster
#' xgb.attr(model_shallow_copy, "my_attr") <- 111
#' xgb.attr(model, "my_attr") # gets modified
#' xgb.attr(model_deep_copy, "my_attr") # doesn't get modified
#' @export
xgb.is.same.Booster <- function(obj1, obj2) {
  if (!inherits(obj1, "xgb.Booster") || !inherits(obj2, "xgb.Booster")) {
    stop("'xgb.is.same.Booster' is only applicable to 'xgb.Booster' objects.")
  }
  return(
    .Call(
      XGPointerEqComparison_R,
      xgb.get.handle(obj1),
      xgb.get.handle(obj2)
    )
  )
}

#' @title Print xgb.Booster
#' @description Print information about `xgb.Booster`.
#' @param x An `xgb.Booster` object.
#' @param ... Not used.
#' @return The same `x` object, returned invisibly
#' @examples
#' data(agaricus.train, package = "xgboost")
#' train <- agaricus.train
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(train$data, label = train$label),
#'   max_depth = 2,
#'   eta = 1,
#'   nthread = 2,
#'   nrounds = 2,
#'   objective = "binary:logistic"
#' )
#'
#' attr(bst, "myattr") <- "memo"
#'
#' print(bst)
#'
#' @export
print.xgb.Booster <- function(x, ...) {
  # this lets it error out when the object comes from an earlier R xgboost version
  handle <- xgb.get.handle(x)
  cat('##### xgb.Booster\n')

  R_attrs <- attributes(x)
  if (!is.null(R_attrs$call)) {
    cat('call:\n  ')
    print(R_attrs$call)
  }

  cat('# of features:', xgb.num_feature(x), '\n')
  cat('# of rounds: ', xgb.get.num.boosted.rounds(x), '\n')

  attr_names <- .Call(XGBoosterGetAttrNames_R, handle)
  if (NROW(attr_names)) {
    cat('xgb.attributes:\n')
    cat("  ", paste(attr_names, collapse = ", "), "\n")
  }

  additional_attr <- setdiff(names(R_attrs), .reserved_cb_names)
  if (NROW(additional_attr)) {
    cat("callbacks:\n  ", paste(additional_attr, collapse = ", "), "\n")
  }

  if (!is.null(R_attrs$evaluation_log)) {
    cat('evaluation_log:\n')
    print(R_attrs$evaluation_log, row.names = FALSE, topn = 2)
  }

  return(invisible(x))
}
