#' Construct xgb.DMatrix object
#'
#' Construct an 'xgb.DMatrix' object from a given data source, which can then be passed to functions
#' such as [xgb.train()] or [predict()].
#'
#' Function `xgb.QuantileDMatrix()` will construct a DMatrix with quantization for the histogram
#' method already applied to it, which can be used to reduce memory usage (compared to using a
#' a regular DMatrix first and then creating a quantization out of it) when using the histogram
#' method (`tree_method = "hist"`, which is the default algorithm), but is not usable for the
#' sorted-indices method (`tree_method = "exact"`), nor for the approximate method
#' (`tree_method = "approx"`).
#'
#' @param data Data from which to create a DMatrix, which can then be used for fitting models or
#' for getting predictions out of a fitted model.
#'
#' Supported input types are as follows:
#' - `matrix` objects, with types `numeric`, `integer`, or `logical`.
#' - `data.frame` objects, with columns of types `numeric`, `integer`, `logical`, or `factor`
#'
#' Note that xgboost uses base-0 encoding for categorical types, hence `factor` types (which use base-1
#' encoding') will be converted inside the function call. Be aware that the encoding used for `factor`
#' types is not kept as part of the model, so in subsequent calls to `predict`, it is the user's
#' responsibility to ensure that factor columns have the same levels as the ones from which the DMatrix
#' was constructed.
#'
#' Other column types are not supported.
#' - CSR matrices, as class `dgRMatrix` from package `Matrix`.
#' - CSC matrices, as class `dgCMatrix` from package `Matrix`.
#'
#' These are **not** supported by `xgb.QuantileDMatrix`.
#' - XGBoost's own binary format for DMatrices, as produced by [xgb.DMatrix.save()].
#' - Single-row CSR matrices, as class `dsparseVector` from package `Matrix`, which is interpreted
#'   as a single row (only when making predictions from a fitted model).
#'
#' @param label Label of the training data. For classification problems, should be passed encoded as
#' integers with numeration starting at zero.
#' @param weight Weight for each instance.
#'
#'   Note that, for ranking task, weights are per-group.  In ranking task, one weight
#'   is assigned to each group (not each data point). This is because we
#'   only care about the relative ordering of data points within each group,
#'   so it doesn't make sense to assign weights to individual data points.
#' @param base_margin Base margin used for boosting from existing model.
#'
#'   In the case of multi-output models, one can also pass multi-dimensional base_margin.
#' @param missing A float value to represents missing values in data (not used when creating DMatrix
#'   from text files). It is useful to change when a zero, infinite, or some other
#'   extreme value represents missing values in data.
#' @param silent whether to suppress printing an informational message after loading from a file.
#' @param feature_names Set names for features. Overrides column names in data frame and matrix.
#'
#'   Note: columns are not referenced by name when calling `predict`, so the column order there
#'   must be the same as in the DMatrix construction, regardless of the column names.
#' @param feature_types Set types for features.
#'
#'   If `data` is a `data.frame` and passing `feature_types` is not supplied,
#'   feature types will be deduced automatically from the column types.
#'
#'   Otherwise, one can pass a character vector with the same length as number of columns in `data`,
#'   with the following possible values:
#'   - "c", which represents categorical columns.
#'   - "q", which represents numeric columns.
#'   - "int", which represents integer columns.
#'   - "i", which represents logical (boolean) columns.
#'
#'   Note that, while categorical types are treated differently from the rest for model fitting
#'   purposes, the other types do not influence the generated model, but have effects in other
#'   functionalities such as feature importances.
#'
#'   **Important**: Categorical features, if specified manually through `feature_types`, must
#'   be encoded as integers with numeration starting at zero, and the same encoding needs to be
#'   applied when passing data to [predict()]. Even if passing `factor` types, the encoding will
#'   not be saved, so make sure that `factor` columns passed to `predict` have the same `levels`.
#' @param nthread Number of threads used for creating DMatrix.
#' @param group Group size for all ranking group.
#' @param qid Query ID for data samples, used for ranking.
#' @param label_lower_bound Lower bound for survival training.
#' @param label_upper_bound Upper bound for survival training.
#' @param feature_weights Set feature weights for column sampling.
#' @param data_split_mode Not used yet. This parameter is for distributed training, which is not yet available for the R package.
#' @inheritParams xgb.train
#' @return An 'xgb.DMatrix' object. If calling `xgb.QuantileDMatrix`, it will have additional
#' subclass `xgb.QuantileDMatrix`.
#'
#' @details
#' Note that DMatrix objects are not serializable through R functions such as [saveRDS()] or [save()].
#' If a DMatrix gets serialized and then de-serialized (for example, when saving data in an R session or caching
#' chunks in an Rmd file), the resulting object will not be usable anymore and will need to be reconstructed
#' from the original source of data.
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' ## Keep the number of threads to 1 for examples
#' nthread <- 1
#' data.table::setDTthreads(nthread)
#' dtrain <- with(
#'   agaricus.train, xgb.DMatrix(data, label = label, nthread = nthread)
#' )
#' fname <- file.path(tempdir(), "xgb.DMatrix.data")
#' xgb.DMatrix.save(dtrain, fname)
#' dtrain <- xgb.DMatrix(fname)
#' @export
#' @rdname xgb.DMatrix
xgb.DMatrix <- function(
  data,
  label = NULL,
  weight = NULL,
  base_margin = NULL,
  missing = NA,
  silent = FALSE,
  feature_names = colnames(data),
  feature_types = NULL,
  nthread = NULL,
  group = NULL,
  qid = NULL,
  label_lower_bound = NULL,
  label_upper_bound = NULL,
  feature_weights = NULL,
  data_split_mode = "row",
  ...
) {
  check.deprecation(deprecated_dmatrix_params, match.call(), ...)
  if (!is.null(group) && !is.null(qid)) {
    stop("Either one of 'group' or 'qid' should be NULL")
  }
  if (data_split_mode != "row") {
    stop("'data_split_mode' is not supported yet.")
  }
  nthread <- as.integer(NVL(nthread, -1L))
  if (typeof(data) == "character") {
    if (length(data) > 1) {
      stop(
        "'data' has class 'character' and length ", length(data),
        ".\n  'data' accepts either a numeric matrix or a single filename."
      )
    }
    data <- path.expand(data)
    if (data_split_mode == "row") {
      data_split_mode <- 0L
    } else if (data_split_mode == "col") {
      data_split_mode <- 1L
    } else {
      stop("Passed invalid 'data_split_mode': ", data_split_mode)
    }
    handle <- .Call(XGDMatrixCreateFromURI_R, data, as.integer(silent), data_split_mode)
  } else if (is.matrix(data)) {
    handle <- .Call(
      XGDMatrixCreateFromMat_R, data, missing, nthread
    )
  } else if (inherits(data, "dgCMatrix")) {
    handle <- .Call(
      XGDMatrixCreateFromCSC_R,
      data@p,
      data@i,
      data@x,
      nrow(data),
      missing,
      nthread
    )
  } else if (inherits(data, "dgRMatrix")) {
    handle <- .Call(
      XGDMatrixCreateFromCSR_R,
      data@p,
      data@j,
      data@x,
      ncol(data),
      missing,
      nthread
    )
  } else if (inherits(data, "dsparseVector")) {
    indptr <- c(0L, as.integer(length(data@i)))
    ind <- as.integer(data@i) - 1L
    handle <- .Call(
      XGDMatrixCreateFromCSR_R,
      indptr,
      ind,
      data@x,
      length(data),
      missing,
      nthread
    )
  } else if (is.data.frame(data)) {
    tmp <- .process.df.for.dmatrix(data, feature_types)
    feature_types <- tmp$feature_types
    handle <- .Call(
      XGDMatrixCreateFromDF_R, tmp$lst, missing, nthread
    )
    rm(tmp)
  } else {
    stop("xgb.DMatrix does not support construction from ", typeof(data))
  }

  dmat <- handle
  attributes(dmat) <- list(
    class = "xgb.DMatrix",
    fields = new.env()
  )
  .set.dmatrix.fields(
    dmat = dmat,
    label = label,
    weight = weight,
    base_margin = base_margin,
    feature_names = feature_names,
    feature_types = feature_types,
    group = group,
    qid = qid,
    label_lower_bound = label_lower_bound,
    label_upper_bound = label_upper_bound,
    feature_weights = feature_weights
  )

  return(dmat)
}

.process.df.for.dmatrix <- function(df, feature_types) {
  if (!nrow(df) || !ncol(df)) {
    stop("'data' is an empty data.frame.")
  }
  if (!is.null(feature_types)) {
    if (!is.character(feature_types) || length(feature_types) != ncol(df)) {
      stop(
        "'feature_types' must be a character vector with one entry per column in 'data'."
      )
    }
  } else {
    feature_types <- sapply(df, function(col) {
      if (is.factor(col)) {
        return("c")
      } else if (is.integer(col)) {
        return("int")
      } else if (is.logical(col)) {
        return("i")
      } else {
        if (!is.numeric(col)) {
          stop("Invalid type in dataframe.")
        }
        return("float")
      }
    })
  }

  lst <- lapply(df, function(col) {
    is_factor <- is.factor(col)
    col <- as.numeric(col)
    if (is_factor) {
      col <- col - 1
    }
    return(col)
  })

  return(list(lst = lst, feature_types = feature_types))
}

.set.dmatrix.fields <- function(
  dmat,
  label,
  weight,
  base_margin,
  feature_names,
  feature_types,
  group,
  qid,
  label_lower_bound,
  label_upper_bound,
  feature_weights
) {
  if (!is.null(label)) {
    setinfo(dmat, "label", label)
  }
  if (!is.null(weight)) {
    setinfo(dmat, "weight", weight)
  }
  if (!is.null(base_margin)) {
    setinfo(dmat, "base_margin", base_margin)
  }
  if (!is.null(feature_names)) {
    setinfo(dmat, "feature_name", feature_names)
  }
  if (!is.null(feature_types)) {
    setinfo(dmat, "feature_type", feature_types)
  }
  if (!is.null(group)) {
    setinfo(dmat, "group", group)
  }
  if (!is.null(qid)) {
    setinfo(dmat, "qid", qid)
  }
  if (!is.null(label_lower_bound)) {
    setinfo(dmat, "label_lower_bound", label_lower_bound)
  }
  if (!is.null(label_upper_bound)) {
    setinfo(dmat, "label_upper_bound", label_upper_bound)
  }
  if (!is.null(feature_weights)) {
    setinfo(dmat, "feature_weights", feature_weights)
  }
}

#' @param ref The training dataset that provides quantile information, needed when creating
#' validation/test dataset with [xgb.QuantileDMatrix()]. Supplying the training DMatrix
#' as a reference means that the same quantisation applied to the training data is
#' applied to the validation/test data
#' @param max_bin The number of histogram bin, should be consistent with the training parameter
#'   `max_bin`.
#'
#'   This is only supported when constructing a QuantileDMatrix.
#' @export
#' @rdname xgb.DMatrix
xgb.QuantileDMatrix <- function(
  data,
  label = NULL,
  weight = NULL,
  base_margin = NULL,
  missing = NA,
  feature_names = colnames(data),
  feature_types = NULL,
  nthread = NULL,
  group = NULL,
  qid = NULL,
  label_lower_bound = NULL,
  label_upper_bound = NULL,
  feature_weights = NULL,
  ref = NULL,
  max_bin = NULL
) {
  nthread <- as.integer(NVL(nthread, -1L))
  if (!is.null(ref) && !inherits(ref, "xgb.DMatrix")) {
    stop("'ref' must be an xgb.DMatrix object.")
  }

  # Note: when passing an integer matrix, it won't get casted to numeric.
  # Since 'int' values as understood by languages like C cannot have missing values,
  # R represents missingness there by assigning them a value equal to the minimum
  # integer. The 'missing' value here is set before the data, so in case of integers,
  # need to make the conversion manually beforehand.
  if (is.matrix(data) && storage.mode(data) %in% c("integer", "logical") && is.na(missing)) {
    missing <- .Call(XGGetRNAIntAsDouble)
  }

  iterator_env <- as.environment(
    list(
      data = data,
      label = label,
      weight = weight,
      base_margin = base_margin,
      missing = missing,
      feature_names = feature_names,
      feature_types = feature_types,
      group = group,
      qid = qid,
      label_lower_bound = label_lower_bound,
      label_upper_bound = label_upper_bound,
      feature_weights = feature_weights
    )
  )
  data_iterator <- .single.data.iterator(iterator_env)

  env_keep_alive <- new.env()
  env_keep_alive$keepalive <- NULL

  # Note: the ProxyDMatrix has its finalizer assigned in the R externalptr
  # object, but that finalizer will only be called once the object is
  # garbage-collected, which doesn't happen immediately after it goes out
  # of scope, hence this piece of code to tigger its destruction earlier
  # and free memory right away.
  proxy_handle <- .make.proxy.handle()
  on.exit({
    .Call(XGDMatrixFree_R, proxy_handle)
  })
  iterator_next <- function() {
    return(xgb.ProxyDMatrix(proxy_handle, data_iterator, env_keep_alive))
  }
  iterator_reset <- function() {
    env_keep_alive$keepalive <- NULL
    return(data_iterator$f_reset(iterator_env))
  }
  calling_env <- environment()

  dmat <- .Call(
    XGQuantileDMatrixCreateFromCallback_R,
    iterator_next,
    iterator_reset,
    calling_env,
    proxy_handle,
    nthread,
    missing,
    max_bin,
    ref
  )
  attributes(dmat) <- list(
    class = c("xgb.DMatrix", "xgb.QuantileDMatrix"),
    fields = attributes(proxy_handle)$fields
  )
  return(dmat)
}

#' XGBoost Data Iterator
#'
#' @description
#' Interface to create a custom data iterator in order to construct a DMatrix
#' from external memory.
#'
#' This function is responsible for generating an R object structure containing callback
#' functions and an environment shared with them.
#'
#' The output structure from this function is then meant to be passed to [xgb.ExtMemDMatrix()],
#' which will consume the data and create a DMatrix from it by executing the callback functions.
#'
#' For more information, and for a usage example, see the documentation for [xgb.ExtMemDMatrix()].
#'
#' @param env An R environment to pass to the callback functions supplied here, which can be
#'   used to keep track of variables to determine how to handle the batches.
#'
#'   For example, one might want to keep track of an iteration number in this environment in order
#'   to know which part of the data to pass next.
#' @param f_next `function(env)` which is responsible for:
#'   - Accessing or retrieving the next batch of data in the iterator.
#'   - Supplying this data by calling function [xgb.DataBatch()] on it and returning the result.
#'   - Keeping track of where in the iterator batch it is or will go next, which can for example
#'     be done by modifiying variables in the `env` variable that is passed here.
#'   - Signaling whether there are more batches to be consumed or not, by returning `NULL`
#'     when the stream of data ends (all batches in the iterator have been consumed), or the result from
#'     calling [xgb.DataBatch()] when there are more batches in the line to be consumed.
#' @param f_reset `function(env)` which is responsible for reseting the data iterator
#'   (i.e. taking it back to the first batch, called before and after the sequence of batches
#'   has been consumed).
#'
#'   Note that, after resetting the iterator, the batches will be accessed again, so the same data
#'   (and in the same order) must be passed in subsequent iterations.
#' @return An `xgb.DataIter` object, containing the same inputs supplied here, which can then
#'   be passed to [xgb.ExtMemDMatrix()].
#' @seealso [xgb.ExtMemDMatrix()], [xgb.DataBatch()].
#' @export
xgb.DataIter <- function(env = new.env(), f_next, f_reset) {
  if (!is.function(f_next)) {
    stop("'f_next' must be a function.")
  }
  if (!is.function(f_reset)) {
    stop("'f_reset' must be a function.")
  }
  out <- list(
    env = env,
    f_next = f_next,
    f_reset = f_reset
  )
  class(out) <- "xgb.DataIter"
  return(out)
}

.qdm.single.fnext <- function(env) {
  curr_iter <- env[["iter"]]
  if (curr_iter >= 1L) {
    return(NULL)
  }

  on.exit({
    env[["iter"]] <- curr_iter + 1L
  })
  return(
    xgb.DataBatch(
      data = env[["data"]],
      label = env[["label"]],
      weight = env[["weight"]],
      base_margin = env[["base_margin"]],
      feature_names = env[["feature_names"]],
      feature_types = env[["feature_types"]],
      group = env[["group"]],
      qid = env[["qid"]],
      label_lower_bound = env[["label_lower_bound"]],
      label_upper_bound = env[["label_upper_bound"]],
      feature_weights = env[["feature_weights"]]
    )
  )
}

.qdm.single.freset <- function(env) {
  env[["iter"]] <- 0L
  return(invisible(NULL))
}

.single.data.iterator <- function(env) {
  env[["iter"]] <- 0L
  return(xgb.DataIter(env, .qdm.single.fnext, .qdm.single.freset))
}

# Only for internal usage
.make.proxy.handle <- function() {
  out <- .Call(XGProxyDMatrixCreate_R)
  attributes(out) <- list(
    class = c("xgb.DMatrix", "xgb.ProxyDMatrix"),
    fields = new.env()
  )
  return(out)
}

#' Structure for Data Batches
#'
#' @description
#' Helper function to supply data in batches of a data iterator when
#' constructing a DMatrix from external memory through [xgb.ExtMemDMatrix()]
#' or through [xgb.QuantileDMatrix.from_iterator()].
#'
#' This function is **only** meant to be called inside of a callback function (which
#' is passed as argument to function [xgb.DataIter()] to construct a data iterator)
#' when constructing a DMatrix through external memory - otherwise, one should call
#' [xgb.DMatrix()] or [xgb.QuantileDMatrix()].
#'
#' The object that results from calling this function directly is **not** like
#' an `xgb.DMatrix` - i.e. cannot be used to train a model, nor to get predictions - only
#' possible usage is to supply data to an iterator, from which a DMatrix is then constructed.
#'
#' For more information and for example usage, see the documentation for [xgb.ExtMemDMatrix()].
#' @inheritParams xgb.DMatrix
#' @param data Batch of data belonging to this batch.
#'
#'   Note that not all of the input types supported by [xgb.DMatrix()] are possible
#'   to pass here. Supported types are:
#'   - `matrix`, with types `numeric`, `integer`, and `logical`. Note that for types
#'     `integer` and `logical`, missing values might not be automatically recognized as
#'     as such - see the documentation for parameter `missing` in [xgb.ExtMemDMatrix()]
#'     for details on this.
#'   - `data.frame`, with the same types as supported by 'xgb.DMatrix' and same
#'     conversions applied to it. See the documentation for parameter `data` in
#'     [xgb.DMatrix()] for details on it.
#'   - CSR matrices, as class `dgRMatrix` from package "Matrix".
#' @return An object of class `xgb.DataBatch`, which is just a list containing the
#'   data and parameters passed here. It does **not** inherit from `xgb.DMatrix`.
#' @seealso [xgb.DataIter()], [xgb.ExtMemDMatrix()].
#' @export
xgb.DataBatch <- function(
  data,
  label = NULL,
  weight = NULL,
  base_margin = NULL,
  feature_names = colnames(data),
  feature_types = NULL,
  group = NULL,
  qid = NULL,
  label_lower_bound = NULL,
  label_upper_bound = NULL,
  feature_weights = NULL
) {
  stopifnot(inherits(data, c("matrix", "data.frame", "dgRMatrix")))
  out <- list(
    data = data,
    label = label,
    weight = weight,
    base_margin = base_margin,
    feature_names = feature_names,
    feature_types = feature_types,
    group = group,
    qid = qid,
    label_lower_bound = label_lower_bound,
    label_upper_bound = label_upper_bound,
    feature_weights = feature_weights
  )
  class(out) <- "xgb.DataBatch"
  return(out)
}

# This is only for internal usage, class is not exposed to the user.
xgb.ProxyDMatrix <- function(proxy_handle, data_iterator, env_keep_alive) {
  env_keep_alive$keepalive <- NULL
  lst <- data_iterator$f_next(data_iterator$env)
  if (is.null(lst)) {
    return(0L)
  }
  if (!inherits(lst, "xgb.DataBatch")) {
    stop("DataIter 'f_next' must return either NULL or the result from calling 'xgb.DataBatch'.")
  }

  if (!is.null(lst$group) && !is.null(lst$qid)) {
    stop("Either one of 'group' or 'qid' should be NULL")
  }
  if (is.data.frame(lst$data)) {
    data <- lst$data
    lst$data <- NULL
    tmp <- .process.df.for.dmatrix(data, lst$feature_types)
    lst$feature_types <- tmp$feature_types
    data <- NULL
    env_keep_alive$keepalive <- tmp
    .Call(XGProxyDMatrixSetDataColumnar_R, proxy_handle, tmp$lst)
  } else if (is.matrix(lst$data)) {
    env_keep_alive$keepalive <- lst
    .Call(XGProxyDMatrixSetDataDense_R, proxy_handle, lst$data)
  } else if (inherits(lst$data, "dgRMatrix")) {
    tmp <- list(p = lst$data@p, j = lst$data@j, x = lst$data@x, ncol = ncol(lst$data))
    env_keep_alive$keepalive <- tmp
    .Call(XGProxyDMatrixSetDataCSR_R, proxy_handle, tmp)
  } else {
    stop("'data' has unsupported type.")
  }

  .set.dmatrix.fields(
    dmat = proxy_handle,
    label = lst$label,
    weight = lst$weight,
    base_margin = lst$base_margin,
    feature_names = lst$feature_names,
    feature_types = lst$feature_types,
    group = lst$group,
    qid = lst$qid,
    label_lower_bound = lst$label_lower_bound,
    label_upper_bound = lst$label_upper_bound,
    feature_weights = lst$feature_weights
  )

  return(1L)
}

#' DMatrix from External Data
#'
#' @description
#' Create a special type of XGBoost 'DMatrix' object from external data
#' supplied by an [xgb.DataIter()] object, potentially passed in batches from a
#' bigger set that might not fit entirely in memory.
#'
#' The data supplied by the iterator is accessed on-demand as needed, multiple times,
#' without being concatenated, but note that fields like 'label' **will** be
#' concatenated from multiple calls to the data iterator.
#'
#' For more information, see the guide 'Using XGBoost External Memory Version':
#' \url{https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html}
#' @details
#' Be aware that construction of external data DMatrices \bold{will cache data on disk}
#' in a compressed format, under the path supplied in `cache_prefix`.
#'
#' External data is not supported for the exact tree method.
#' @inheritParams xgb.DMatrix
#' @param data_iterator A data iterator structure as returned by [xgb.DataIter()],
#'   which includes an environment shared between function calls, and functions to access
#'   the data in batches on-demand.
#' @param cache_prefix The path of cache file, caller must initialize all the directories in this path.
#' @param missing A float value to represents missing values in data.
#'
#'   Note that, while functions like [xgb.DMatrix()] can take a generic `NA` and interpret it
#'   correctly for different types like `numeric` and `integer`, if an `NA` value is passed here,
#'   it will not be adapted for different input types.
#'
#'   For example, in R `integer` types, missing values are represented by integer number `-2147483648`
#'   (since machine 'integer' types do not have an inherent 'NA' value) - hence, if one passes `NA`,
#'   which is interpreted as a floating-point NaN by [xgb.ExtMemDMatrix()] and by
#'   [xgb.QuantileDMatrix.from_iterator()], these integer missing values will not be treated as missing.
#'   This should not pose any problem for `numeric` types, since they do have an inheret NaN value.
#' @return An 'xgb.DMatrix' object, with subclass 'xgb.ExtMemDMatrix', in which the data is not
#'   held internally but accessed through the iterator when needed.
#' @seealso [xgb.DataIter()], [xgb.DataBatch()], [xgb.QuantileDMatrix.from_iterator()]
#' @examples
#' data(mtcars)
#'
#' # This custom environment will be passed to the iterator
#' # functions at each call. It is up to the user to keep
#' # track of the iteration number in this environment.
#' iterator_env <- as.environment(
#'   list(
#'     iter = 0,
#'     x = mtcars[, -1],
#'     y = mtcars[, 1]
#'   )
#' )
#'
#' # Data is passed in two batches.
#' # In this example, batches are obtained by subsetting the 'x' variable.
#' # This is not advantageous to do, since the data is already loaded in memory
#' # and can be passed in full in one go, but there can be situations in which
#' # only a subset of the data will fit in the computer's memory, and it can
#' # be loaded in batches that are accessed one-at-a-time only.
#' iterator_next <- function(iterator_env) {
#'   curr_iter <- iterator_env[["iter"]]
#'   if (curr_iter >= 2) {
#'     # there are only two batches, so this signals end of the stream
#'     return(NULL)
#'   }
#'
#'   if (curr_iter == 0) {
#'     x_batch <- iterator_env[["x"]][1:16, ]
#'     y_batch <- iterator_env[["y"]][1:16]
#'   } else {
#'     x_batch <- iterator_env[["x"]][17:32, ]
#'     y_batch <- iterator_env[["y"]][17:32]
#'   }
#'   on.exit({
#'     iterator_env[["iter"]] <- curr_iter + 1
#'   })
#'
#'   # Function 'xgb.DataBatch' must be called manually
#'   # at each batch with all the appropriate attributes,
#'   # such as feature names and feature types.
#'   return(xgb.DataBatch(data = x_batch, label = y_batch))
#' }
#'
#' # This moves the iterator back to its beginning
#' iterator_reset <- function(iterator_env) {
#'   iterator_env[["iter"]] <- 0
#' }
#'
#' data_iterator <- xgb.DataIter(
#'   env = iterator_env,
#'   f_next = iterator_next,
#'   f_reset = iterator_reset
#' )
#' cache_prefix <- tempdir()
#'
#' # DMatrix will be constructed from the iterator's batches
#' dm <- xgb.ExtMemDMatrix(data_iterator, cache_prefix, nthread = 1)
#'
#' # After construction, can be used as a regular DMatrix
#' params <- xgb.params(nthread = 1, objective = "reg:squarederror")
#' model <- xgb.train(data = dm, nrounds = 2, params = params)
#'
#' # Predictions can also be called on it, and should be the same
#' # as if the data were passed differently.
#' pred_dm <- predict(model, dm)
#' pred_mat <- predict(model, as.matrix(mtcars[, -1]))
#' @export
xgb.ExtMemDMatrix <- function(
  data_iterator,
  cache_prefix = tempdir(),
  missing = NA,
  nthread = NULL
) {
  stopifnot(inherits(data_iterator, "xgb.DataIter"))
  stopifnot(is.character(cache_prefix))

  cache_prefix <- path.expand(cache_prefix)
  nthread <- as.integer(NVL(nthread, -1L))

  # The purpose of this environment is to keep data alive (protected from the
  # garbage collector) after setting the data in the proxy dmatrix. The data
  # held here (under name 'keepalive') should be unset (leaving it unprotected
  # for garbage collection) before the start of each data iteration batch and
  # during each iterator reset.
  env_keep_alive <- new.env()
  env_keep_alive$keepalive <- NULL

  proxy_handle <- .make.proxy.handle()
  on.exit({
    .Call(XGDMatrixFree_R, proxy_handle)
  })
  iterator_next <- function() {
    return(xgb.ProxyDMatrix(proxy_handle, data_iterator, env_keep_alive))
  }
  iterator_reset <- function() {
    env_keep_alive$keepalive <- NULL
    return(data_iterator$f_reset(data_iterator$env))
  }
  calling_env <- environment()

  dmat <- .Call(
    XGDMatrixCreateFromCallback_R,
    iterator_next,
    iterator_reset,
    calling_env,
    proxy_handle,
    nthread,
    missing,
    cache_prefix
  )

  attributes(dmat) <- list(
    class = c("xgb.DMatrix", "xgb.ExtMemDMatrix"),
    fields = attributes(proxy_handle)$fields
  )
  return(dmat)
}


#' QuantileDMatrix from External Data
#'
#' @description
#' Create an `xgb.QuantileDMatrix` object (exact same class as would be returned by
#' calling function [xgb.QuantileDMatrix()], with the same advantages and limitations) from
#' external data supplied by [xgb.DataIter()], potentially passed in batches from
#' a bigger set that might not fit entirely in memory, same way as [xgb.ExtMemDMatrix()].
#'
#' Note that, while external data will only be loaded through the iterator (thus the full data
#' might not be held entirely in-memory), the quantized representation of the data will get
#' created in-memory, being concatenated from multiple calls to the data iterator. The quantized
#' version is typically lighter than the original data, so there might be cases in which this
#' representation could potentially fit in memory even if the full data does not.
#'
#' For more information, see the guide 'Using XGBoost External Memory Version':
#' \url{https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html}
#' @inheritParams xgb.ExtMemDMatrix
#' @inheritParams xgb.QuantileDMatrix
#' @return An 'xgb.DMatrix' object, with subclass 'xgb.QuantileDMatrix'.
#' @seealso [xgb.DataIter()], [xgb.DataBatch()], [xgb.ExtMemDMatrix()],
#' [xgb.QuantileDMatrix()]
#' @export
xgb.QuantileDMatrix.from_iterator <- function( # nolint
  data_iterator,
  missing = NA,
  nthread = NULL,
  ref = NULL,
  max_bin = NULL
) {
  stopifnot(inherits(data_iterator, "xgb.DataIter"))
  if (!is.null(ref) && !inherits(ref, "xgb.DMatrix")) {
    stop("'ref' must be an xgb.DMatrix object.")
  }

  nthread <- as.integer(NVL(nthread, -1L))

  env_keep_alive <- new.env()
  env_keep_alive$keepalive <- NULL
  proxy_handle <- .make.proxy.handle()
  on.exit({
    .Call(XGDMatrixFree_R, proxy_handle)
  })
  iterator_next <- function() {
    return(xgb.ProxyDMatrix(proxy_handle, data_iterator, env_keep_alive))
  }
  iterator_reset <- function() {
    env_keep_alive$keepalive <- NULL
    return(data_iterator$f_reset(data_iterator$env))
  }
  calling_env <- environment()

  dmat <- .Call(
    XGQuantileDMatrixCreateFromCallback_R,
    iterator_next,
    iterator_reset,
    calling_env,
    proxy_handle,
    nthread,
    missing,
    max_bin,
    ref
  )

  attributes(dmat) <- list(
    class = c("xgb.DMatrix", "xgb.QuantileDMatrix"),
    fields = attributes(proxy_handle)$fields
  )
  return(dmat)
}

#' Check whether DMatrix object has a field
#'
#' Checks whether an xgb.DMatrix object has a given field assigned to
#' it, such as weights, labels, etc.
#' @param object The DMatrix object to check for the given `info` field.
#' @param info The field to check for presence or absence in `object`.
#' @seealso [xgb.DMatrix()], [getinfo.xgb.DMatrix()], [setinfo.xgb.DMatrix()]
#' @examples
#' x <- matrix(1:10, nrow = 5)
#' dm <- xgb.DMatrix(x, nthread = 1)
#'
#' # 'dm' so far does not have any fields set
#' xgb.DMatrix.hasinfo(dm, "label")
#'
#' # Fields can be added after construction
#' setinfo(dm, "label", 1:5)
#' xgb.DMatrix.hasinfo(dm, "label")
#' @export
xgb.DMatrix.hasinfo <- function(object, info) {
  if (!inherits(object, "xgb.DMatrix")) {
    stop("Object is not an 'xgb.DMatrix'.")
  }
  if (.Call(XGCheckNullPtr_R, object)) {
    warning("xgb.DMatrix object is invalid. Must be constructed again.")
    return(FALSE)
  }
  return(NVL(attr(object, "fields")[[info]], FALSE))
}


#' Dimensions of xgb.DMatrix
#'
#' Returns a vector of numbers of rows and of columns in an `xgb.DMatrix`.
#'
#' @param x Object of class `xgb.DMatrix`
#'
#' @details
#' Note: since [nrow()] and [ncol()] internally use [dim()], they can also
#' be directly used with an `xgb.DMatrix` object.
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label = train$label, nthread = 2)
#'
#' stopifnot(nrow(dtrain) == nrow(train$data))
#' stopifnot(ncol(dtrain) == ncol(train$data))
#' stopifnot(all(dim(dtrain) == dim(train$data)))
#'
#' @export
dim.xgb.DMatrix <- function(x) {
  c(.Call(XGDMatrixNumRow_R, x), .Call(XGDMatrixNumCol_R, x))
}


#' Handling of column names of `xgb.DMatrix`
#'
#' Only column names are supported for `xgb.DMatrix`, thus setting of
#' row names would have no effect and returned row names would be `NULL`.
#'
#' @param x Object of class `xgb.DMatrix`.
#' @param value A list of two elements: the first one is ignored
#'   and the second one is column names
#'
#' @details
#' Generic [dimnames()] methods are used by [colnames()].
#' Since row names are irrelevant, it is recommended to use [colnames()] directly.
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label = train$label, nthread = 2)
#' dimnames(dtrain)
#' colnames(dtrain)
#' colnames(dtrain) <- make.names(1:ncol(train$data))
#' print(dtrain, verbose = TRUE)
#'
#' @rdname dimnames.xgb.DMatrix
#' @export
dimnames.xgb.DMatrix <- function(x) {
  fn <- getinfo(x, "feature_name")
  ## row names is null.
  list(NULL, fn)
}

#' @rdname dimnames.xgb.DMatrix
#' @export
`dimnames<-.xgb.DMatrix` <- function(x, value) {
  if (!is.list(value) || length(value) != 2L)
    stop("invalid 'dimnames' given: must be a list of two elements")
  if (!is.null(value[[1L]]))
    stop("xgb.DMatrix does not have rownames")
  if (is.null(value[[2]])) {
    setinfo(x, "feature_name", NULL)
    return(x)
  }
  if (ncol(x) != length(value[[2]])) {
    stop("can't assign ", length(value[[2]]), " colnames to a ", ncol(x), " column xgb.DMatrix")
  }
  setinfo(x, "feature_name", value[[2]])
  x
}


#' Get or set information of xgb.DMatrix and xgb.Booster objects
#'
#' @param object Object of class `xgb.DMatrix` or `xgb.Booster`.
#' @param name The name of the information field to get (see details).
#' @return For `getinfo()`, will return the requested field. For `setinfo()`,
#'   will always return value `TRUE` if it succeeds.
#' @details
#' The `name` field can be one of the following for `xgb.DMatrix`:
#' - label
#' - weight
#' - base_margin
#' - label_lower_bound
#' - label_upper_bound
#' - group
#' - feature_type
#' - feature_name
#' - nrow
#'
#' See the documentation for [xgb.DMatrix()] for more information about these fields.
#'
#' For `xgb.Booster`, can be one of the following:
#' - `feature_type`
#' - `feature_name`
#'
#' Note that, while 'qid' cannot be retrieved, it is possible to get the equivalent 'group'
#' for a DMatrix that had 'qid' assigned.
#'
#' **Important**: when calling [setinfo()], the objects are modified in-place. See
#' [xgb.copy.Booster()] for an idea of this in-place assignment works.
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#'
#' labels <- getinfo(dtrain, "label")
#' setinfo(dtrain, "label", 1 - labels)
#'
#' labels2 <- getinfo(dtrain, "label")
#' stopifnot(all(labels2 == 1 - labels))
#' @rdname getinfo
#' @export
getinfo <- function(object, name) UseMethod("getinfo")

#' @rdname getinfo
#' @export
getinfo.xgb.DMatrix <- function(object, name) {
  allowed_int_fields <- 'group'
  allowed_float_fields <- c(
    'label', 'weight', 'base_margin',
    'label_lower_bound', 'label_upper_bound'
  )
  allowed_str_fields <- c("feature_type", "feature_name")
  allowed_fields <- c(allowed_float_fields, allowed_int_fields, allowed_str_fields, 'nrow')

  if (typeof(name) != "character" ||
        length(name) != 1 ||
        !name %in% allowed_fields) {
    stop("getinfo: name must be one of the following\n",
         paste(paste0("'", allowed_fields, "'"), collapse = ", "))
  }
  if (name == "nrow") {
    ret <- nrow(object)
  } else if (name %in% allowed_str_fields) {
    ret <- .Call(XGDMatrixGetStrFeatureInfo_R, object, name)
  } else if (name %in% allowed_float_fields) {
    ret <- .Call(XGDMatrixGetFloatInfo_R, object, name)
    if (length(ret) > nrow(object)) {
      ret <- matrix(ret, nrow = nrow(object), byrow = TRUE)
    }
  } else if (name %in% allowed_int_fields) {
    if (name == "group") {
      name <- "group_ptr"
    }
    ret <- .Call(XGDMatrixGetUIntInfo_R, object, name)
    if (length(ret) > nrow(object)) {
      ret <- matrix(ret, nrow = nrow(object), byrow = TRUE)
    }
  }
  if (length(ret) == 0) return(NULL)
  return(ret)
}

#' @rdname getinfo
#' @param info The specific field of information to set.
#'
#' @details
#' See the documentation for [xgb.DMatrix()] for possible fields that can be set
#' (which correspond to arguments in that function).
#'
#' Note that the following fields are allowed in the construction of an `xgb.DMatrix`
#' but **are not** allowed here:
#' - data
#' - missing
#' - silent
#' - nthread
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#'
#' labels <- getinfo(dtrain, "label")
#' setinfo(dtrain, "label", 1 - labels)
#'
#' labels2 <- getinfo(dtrain, "label")
#' stopifnot(all.equal(labels2, 1 - labels))
#' @export
setinfo <- function(object, name, info) UseMethod("setinfo")

#' @rdname getinfo
#' @export
setinfo.xgb.DMatrix <- function(object, name, info) {
  .internal.setinfo.xgb.DMatrix(object, name, info)
  attr(object, "fields")[[name]] <- TRUE
  return(TRUE)
}

.internal.setinfo.xgb.DMatrix <- function(object, name, info) {
  if (name == "label") {
    if (NROW(info) != nrow(object))
      stop("The length of labels must equal to the number of rows in the input data")
    .Call(XGDMatrixSetInfo_R, object, name, info)
    return(TRUE)
  }
  if (name == "label_lower_bound") {
    if (NROW(info) != nrow(object))
      stop("The length of lower-bound labels must equal to the number of rows in the input data")
    .Call(XGDMatrixSetInfo_R, object, name, info)
    return(TRUE)
  }
  if (name == "label_upper_bound") {
    if (NROW(info) != nrow(object))
      stop("The length of upper-bound labels must equal to the number of rows in the input data")
    .Call(XGDMatrixSetInfo_R, object, name, info)
    return(TRUE)
  }
  if (name == "weight") {
    .Call(XGDMatrixSetInfo_R, object, name, info)
    return(TRUE)
  }
  if (name == "base_margin") {
    .Call(XGDMatrixSetInfo_R, object, name, info)
    return(TRUE)
  }
  if (name == "group") {
    if (sum(info) != nrow(object))
      stop("The sum of groups must equal to the number of rows in the input data")
    .Call(XGDMatrixSetInfo_R, object, name, info)
    return(TRUE)
  }
  if (name == "qid") {
    if (NROW(info) != nrow(object))
      stop("The length of qid assignments must equal to the number of rows in the input data")
    .Call(XGDMatrixSetInfo_R, object, name, info)
    return(TRUE)
  }
  if (name == "feature_weights") {
    if (NROW(info) != ncol(object)) {
      stop("The number of feature weights must equal to the number of columns in the input data")
    }
    .Call(XGDMatrixSetInfo_R, object, name, info)
    return(TRUE)
  }

  set_feat_info <- function(name) {
    msg <- sprintf(
      "The number of %s must equal to the number of columns in the input data. %s vs. %s",
      name,
      length(info),
      ncol(object)
    )
    if (!is.null(info)) {
      info <- as.list(info)
      if (length(info) != ncol(object)) {
        stop(msg)
      }
    }
    .Call(XGDMatrixSetStrFeatureInfo_R, object, name, info)
  }
  if (name == "feature_name") {
    set_feat_info("feature_name")
    return(TRUE)
  }
  if (name == "feature_type") {
    set_feat_info("feature_type")
    return(TRUE)
  }
  stop("setinfo: unknown info name ", name)
}

#' Get Quantile Cuts from DMatrix
#'
#' @description
#' Get the quantile cuts (a.k.a. borders) from an `xgb.DMatrix`
#' that has been quantized for the histogram method (`tree_method = "hist"`).
#'
#' These cuts are used in order to assign observations to bins - i.e. these are ordered
#' boundaries which are used to determine assignment condition `border_low < x < border_high`.
#' As such, the first and last bin will be outside of the range of the data, so as to include
#' all of the observations there.
#'
#' If a given column has 'n' bins, then there will be 'n+1' cuts / borders for that column,
#' which will be output in sorted order from lowest to highest.
#'
#' Different columns can have different numbers of bins according to their range.
#' @param dmat An `xgb.DMatrix` object, as returned by [xgb.DMatrix()].
#' @param output Output format for the quantile cuts. Possible options are:
#'   - "list"` will return the output as a list with one entry per column, where
#'     each column will have a numeric vector with the cuts. The list will be named if
#'     `dmat` has column names assigned to it.
#'   - `"arrays"` will return a list with entries `indptr` (base-0 indexing) and
#'     `data`. Here, the cuts for column 'i' are obtained by slicing 'data' from entries
#' `   indptr[i]+1` to `indptr[i+1]`.
#' @return The quantile cuts, in the format specified by parameter `output`.
#' @examples
#' data(mtcars)
#'
#' y <- mtcars$mpg
#' x <- as.matrix(mtcars[, -1])
#' dm <- xgb.DMatrix(x, label = y, nthread = 1)
#'
#' # DMatrix is not quantized right away, but will be once a hist model is generated
#' model <- xgb.train(
#'   data = dm,
#'   params = xgb.params(tree_method = "hist", max_bin = 8, nthread = 1),
#'   nrounds = 3
#' )
#'
#' # Now can get the quantile cuts
#' xgb.get.DMatrix.qcut(dm)
#' @export
xgb.get.DMatrix.qcut <- function(dmat, output = c("list", "arrays")) { # nolint
  stopifnot(inherits(dmat, "xgb.DMatrix"))
  output <- head(output, 1L)
  stopifnot(output %in% c("list", "arrays"))
  res <- .Call(XGDMatrixGetQuantileCut_R, dmat)
  if (output == "arrays") {
    return(res)
  } else {
    feature_names <- getinfo(dmat, "feature_name")
    ncols <- length(res$indptr) - 1
    out <- lapply(
      seq(1, ncols),
      function(col) {
        st <- res$indptr[col]
        end <- res$indptr[col + 1]
        if (end <= st) {
          return(numeric())
        }
        return(res$data[seq(1 + st, end)])
      }
    )
    if (NROW(feature_names)) {
      names(out) <- feature_names
    }
    return(out)
  }
}

#' Get Number of Non-Missing Entries in DMatrix
#'
#' @param dmat An `xgb.DMatrix` object, as returned by [xgb.DMatrix()].
#' @return The number of non-missing entries in the DMatrix.
#' @export
xgb.get.DMatrix.num.non.missing <- function(dmat) { # nolint
  stopifnot(inherits(dmat, "xgb.DMatrix"))
  return(.Call(XGDMatrixNumNonMissing_R, dmat))
}

#' Get DMatrix Data
#'
#' @param dmat An `xgb.DMatrix` object, as returned by [xgb.DMatrix()].
#' @return The data held in the DMatrix, as a sparse CSR matrix (class `dgRMatrix`
#' from package `Matrix`). If it had feature names, these will be added as column names
#' in the output.
#' @export
xgb.get.DMatrix.data <- function(dmat) {
  stopifnot(inherits(dmat, "xgb.DMatrix"))
  res <- .Call(XGDMatrixGetDataAsCSR_R, dmat)
  out <- methods::new("dgRMatrix")
  nrows <- as.integer(length(res$indptr) - 1)
  out@p <- res$indptr
  out@j <- res$indices
  out@x <- res$data
  out@Dim <- as.integer(c(nrows, res$ncols))

  feature_names <- getinfo(dmat, "feature_name")
  dim_names <- list(NULL, NULL)
  if (NROW(feature_names)) {
    dim_names[[2L]] <- feature_names
  }
  out@Dimnames <- dim_names
  return(out)
}

#' Slice DMatrix
#'
#' Get a new DMatrix containing the specified rows of original xgb.DMatrix object.
#'
#' @param object Object of class `xgb.DMatrix`.
#' @param idxset An integer vector of indices of rows needed (base-1 indexing).
#' @param allow_groups Whether to allow slicing an `xgb.DMatrix` with `group` (or
#'   equivalently `qid`) field. Note that in such case, the result will not have
#'   the groups anymore - they need to be set manually through [setinfo()].
#' @param colset Currently not used (columns subsetting is not available).
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#'
#' dsub <- xgb.slice.DMatrix(dtrain, 1:42)
#' labels1 <- getinfo(dsub, "label")
#'
#' dsub <- dtrain[1:42, ]
#' labels2 <- getinfo(dsub, "label")
#' all.equal(labels1, labels2)
#'
#' @rdname xgb.slice.DMatrix
#' @export
xgb.slice.DMatrix <- function(object, idxset, allow_groups = FALSE) {
  if (!inherits(object, "xgb.DMatrix")) {
    stop("object must be xgb.DMatrix")
  }
  ret <- .Call(XGDMatrixSliceDMatrix_R, object, idxset, allow_groups)

  attr_list <- attributes(object)
  nr <- nrow(object)
  len <- sapply(attr_list, NROW)
  ind <- which(len == nr)
  if (length(ind) > 0) {
    nms <- names(attr_list)[ind]
    for (i in seq_along(ind)) {
      obj_attr <- attr(object, nms[i])
      if (NCOL(obj_attr) > 1) {
        attr(ret, nms[i]) <- obj_attr[idxset, ]
      } else {
        attr(ret, nms[i]) <- obj_attr[idxset]
      }
    }
  }

  out <- structure(ret, class = "xgb.DMatrix")
  parent_fields <- as.list(attributes(object)$fields)
  if (NROW(parent_fields)) {
    child_fields <- parent_fields[!(names(parent_fields) %in% c("group", "qid"))]
    child_fields <- as.environment(child_fields)
    attributes(out)$fields <- child_fields
  }
  return(out)
}

#' @rdname xgb.slice.DMatrix
#' @export
`[.xgb.DMatrix` <- function(object, idxset, colset = NULL) {
  xgb.slice.DMatrix(object, idxset)
}


#' Print xgb.DMatrix
#'
#' Print information about xgb.DMatrix.
#' Currently it displays dimensions and presence of info-fields and colnames.
#'
#' @param x An xgb.DMatrix object.
#' @param verbose Whether to print colnames (when present).
#' @param ... Not currently used.
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#' dtrain
#'
#' print(dtrain, verbose = TRUE)
#'
#' @method print xgb.DMatrix
#' @export
print.xgb.DMatrix <- function(x, verbose = FALSE, ...) {
  if (.Call(XGCheckNullPtr_R, x)) {
    cat("INVALID xgb.DMatrix object. Must be constructed anew.\n")
    return(invisible(x))
  }
  class_print <- if (inherits(x, "xgb.QuantileDMatrix")) {
    "xgb.QuantileDMatrix"
  } else if (inherits(x, "xgb.ExtMemDMatrix")) {
    "xgb.ExtMemDMatrix"
  } else if (inherits(x, "xgb.ProxyDMatrix")) {
    "xgb.ProxyDMatrix"
  } else {
    "xgb.DMatrix"
  }

  cat(class_print, ' dim:', nrow(x), 'x', ncol(x), ' info: ')
  infos <- names(attributes(x)$fields)
  infos <- infos[infos != "feature_name"]
  if (!NROW(infos)) infos <- "NA"
  infos <- infos[order(infos)]
  infos <- paste(infos, collapse = ", ")
  cat(infos)
  cnames <- colnames(x)
  cat('  colnames:')
  if (verbose && !is.null(cnames)) {
    cat("\n'")
    cat(cnames, sep = "','")
    cat("'")
  } else {
    if (is.null(cnames)) cat(' no')
    else cat(' yes')
  }
  cat("\n")
  invisible(x)
}
