#' Construct xgb.DMatrix object
#'
#' Construct xgb.DMatrix object from either a dense matrix, a sparse matrix, or a local file.
#' Supported input file formats are either a LIBSVM text file or a binary file that was created previously by
#' \code{\link{xgb.DMatrix.save}}).
#'
#' @param data a \code{matrix} object (either numeric or integer), a \code{dgCMatrix} object,
#'        a \code{dgRMatrix} object,
#'        a \code{dsparseVector} object (only when making predictions from a fitted model, will be
#'        interpreted as a row vector), or a character string representing a filename.
#' @param label Label of the training data.
#' @param weight Weight for each instance.
#'
#' Note that, for ranking task, weights are per-group.  In ranking task, one weight
#' is assigned to each group (not each data point). This is because we
#' only care about the relative ordering of data points within each group,
#' so it doesn't make sense to assign weights to individual data points.
#' @param base_margin Base margin used for boosting from existing model.
#'
#'        In the case of multi-output models, one can also pass multi-dimensional base_margin.
#' @param missing a float value to represents missing values in data (used only when input is a dense matrix).
#'        It is useful when a 0 or some other extreme value represents missing values in data.
#' @param silent whether to suppress printing an informational message after loading from a file.
#' @param feature_names Set names for features. Overrides column names in data
#'        frame and matrix.
#' @param nthread Number of threads used for creating DMatrix.
#' @param group Group size for all ranking group.
#' @param qid Query ID for data samples, used for ranking.
#' @param label_lower_bound Lower bound for survival training.
#' @param label_upper_bound Upper bound for survival training.
#' @param feature_weights Set feature weights for column sampling.
#' @param enable_categorical Experimental support of specializing for categorical features.
#'
#'                           If passing 'TRUE' and 'data' is a data frame,
#'                           columns of categorical types will automatically
#'                           be set to be of categorical type (feature_type='c') in the resulting DMatrix.
#'
#'                           If passing 'FALSE' and 'data' is a data frame with categorical columns,
#'                           it will result in an error being thrown.
#'
#'                           If 'data' is not a data frame, this argument is ignored.
#'
#'                           JSON/UBJSON serialization format is required for this.
#'
#' @details
#' Note that DMatrix objects are not serializable through R functions such as \code{saveRDS} or \code{save}.
#' If a DMatrix gets serialized and then de-serialized (for example, when saving data in an R session or caching
#' chunks in an Rmd file), the resulting object will not be usable anymore and will need to be reconstructed
#' from the original source of data.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
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
xgb.DMatrix <- function(
  data,
  label = NULL,
  weight = NULL,
  base_margin = NULL,
  missing = NA,
  silent = FALSE,
  feature_names = colnames(data),
  nthread = NULL,
  group = NULL,
  qid = NULL,
  label_lower_bound = NULL,
  label_upper_bound = NULL,
  feature_weights = NULL,
  enable_categorical = FALSE
) {
  if (!is.null(group) && !is.null(qid)) {
    stop("Either one of 'group' or 'qid' should be NULL")
  }
  ctypes <- NULL
  if (typeof(data) == "character") {
    if (length(data) > 1) {
      stop(
        "'data' has class 'character' and length ", length(data),
        ".\n  'data' accepts either a numeric matrix or a single filename."
      )
    }
    data <- path.expand(data)
    handle <- .Call(XGDMatrixCreateFromFile_R, data, as.integer(silent))
  } else if (is.matrix(data)) {
    handle <- .Call(
      XGDMatrixCreateFromMat_R, data, missing, as.integer(NVL(nthread, -1))
    )
  } else if (inherits(data, "dgCMatrix")) {
    handle <- .Call(
      XGDMatrixCreateFromCSC_R,
      data@p,
      data@i,
      data@x,
      nrow(data),
      missing,
      as.integer(NVL(nthread, -1))
    )
  } else if (inherits(data, "dgRMatrix")) {
    handle <- .Call(
      XGDMatrixCreateFromCSR_R,
      data@p,
      data@j,
      data@x,
      ncol(data),
      missing,
      as.integer(NVL(nthread, -1))
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
      as.integer(NVL(nthread, -1))
    )
  } else if (is.data.frame(data)) {
    ctypes <- sapply(data, function(x) {
      if (is.factor(x)) {
        if (!enable_categorical) {
          stop(
            "When factor type is used, the parameter `enable_categorical`",
            " must be set to TRUE."
          )
        }
        "c"
      } else if (is.integer(x)) {
        "int"
      } else if (is.logical(x)) {
        "i"
      } else {
        if (!is.numeric(x)) {
          stop("Invalid type in dataframe.")
        }
        "float"
      }
    })
    ## as.data.frame somehow converts integer/logical into real.
    data <- as.data.frame(sapply(data, function(x) {
      if (is.factor(x)) {
        ## XGBoost uses 0-based indexing.
        as.numeric(x) - 1
      } else {
        x
      }
    }))
    handle <- .Call(
      XGDMatrixCreateFromDF_R, data, missing, as.integer(NVL(nthread, -1))
    )
  } else {
    stop("xgb.DMatrix does not support construction from ", typeof(data))
  }

  dmat <- handle
  attributes(dmat) <- list(
    class = "xgb.DMatrix",
    fields = new.env()
  )

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
  if (!is.null(ctypes)) {
    setinfo(dmat, "feature_type", ctypes)
  }

  return(dmat)
}

#' @title Check whether DMatrix object has a field
#' @description Checks whether an xgb.DMatrix object has a given field assigned to
#' it, such as weights, labels, etc.
#' @param object The DMatrix object to check for the given \code{info} field.
#' @param info The field to check for presence or absence in \code{object}.
#' @seealso \link{xgb.DMatrix}, \link{getinfo.xgb.DMatrix}, \link{setinfo.xgb.DMatrix}
#' @examples
#' library(xgboost)
#' x <- matrix(1:10, nrow = 5)
#' dm <- xgb.DMatrix(x, nthread = 1)
#'
#' # 'dm' so far doesn't have any fields set
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


# get dmatrix from data, label
# internal helper method
xgb.get.DMatrix <- function(data, label, missing, weight, nthread) {
  if (inherits(data, "dgCMatrix") || is.matrix(data)) {
    if (is.null(label)) {
      stop("label must be provided when data is a matrix")
    }
    dtrain <- xgb.DMatrix(data, label = label, missing = missing, nthread = nthread)
    if (!is.null(weight)) {
      setinfo(dtrain, "weight", weight)
    }
  } else {
    if (!is.null(label)) {
      warning("xgboost: label will be ignored.")
    }
    if (is.character(data)) {
      data <- path.expand(data)
      dtrain <- xgb.DMatrix(data[1])
    } else if (inherits(data, "xgb.DMatrix")) {
      dtrain <- data
    } else if (inherits(data, "data.frame")) {
      stop("xgboost doesn't support data.frame as input. Convert it to matrix first.")
    } else {
      stop("xgboost: invalid input data")
    }
  }
  return(dtrain)
}


#' Dimensions of xgb.DMatrix
#'
#' Returns a vector of numbers of rows and of columns in an \code{xgb.DMatrix}.
#' @param x Object of class \code{xgb.DMatrix}
#'
#' @details
#' Note: since \code{nrow} and \code{ncol} internally use \code{dim}, they can also
#' be directly used with an \code{xgb.DMatrix} object.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label, nthread = 2)
#'
#' stopifnot(nrow(dtrain) == nrow(train$data))
#' stopifnot(ncol(dtrain) == ncol(train$data))
#' stopifnot(all(dim(dtrain) == dim(train$data)))
#'
#' @export
dim.xgb.DMatrix <- function(x) {
  c(.Call(XGDMatrixNumRow_R, x), .Call(XGDMatrixNumCol_R, x))
}


#' Handling of column names of \code{xgb.DMatrix}
#'
#' Only column names are supported for \code{xgb.DMatrix}, thus setting of
#' row names would have no effect and returned row names would be NULL.
#'
#' @param x object of class \code{xgb.DMatrix}
#' @param value a list of two elements: the first one is ignored
#'        and the second one is column names
#'
#' @details
#' Generic \code{dimnames} methods are used by \code{colnames}.
#' Since row names are irrelevant, it is recommended to use \code{colnames} directly.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label, nthread = 2)
#' dimnames(dtrain)
#' colnames(dtrain)
#' colnames(dtrain) <- make.names(1:ncol(train$data))
#' print(dtrain, verbose=TRUE)
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


#' @title Get or set information of xgb.DMatrix and xgb.Booster objects
#' @param object Object of class \code{xgb.DMatrix} of `xgb.Booster`.
#' @param name the name of the information field to get (see details)
#' @return For `getinfo`, will return the requested field. For `setinfo`, will always return value `TRUE`
#' if it succeeds.
#' @details
#' The \code{name} field can be one of the following for `xgb.DMatrix`:
#'
#' \itemize{
#'     \item \code{label}
#'     \item \code{weight}
#'     \item \code{base_margin}
#'     \item \code{label_lower_bound}
#'     \item \code{label_upper_bound}
#'     \item \code{group}
#'     \item \code{feature_type}
#'     \item \code{feature_name}
#'     \item \code{nrow}
#' }
#' See the documentation for \link{xgb.DMatrix} for more information about these fields.
#'
#' For `xgb.Booster`, can be one of the following:
#' \itemize{
#'     \item \code{feature_type}
#'     \item \code{feature_name}
#' }
#'
#' Note that, while 'qid' cannot be retrieved, it's possible to get the equivalent 'group'
#' for a DMatrix that had 'qid' assigned.
#'
#' \bold{Important}: when calling `setinfo`, the objects are modified in-place. See
#' \link{xgb.copy.Booster} for an idea of this in-place assignment works.
#' @examples
#' data(agaricus.train, package='xgboost')
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#'
#' labels <- getinfo(dtrain, 'label')
#' setinfo(dtrain, 'label', 1-labels)
#'
#' labels2 <- getinfo(dtrain, 'label')
#' stopifnot(all(labels2 == 1-labels))
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
#' @param info the specific field of information to set
#'
#' @details
#' See the documentation for \link{xgb.DMatrix} for possible fields that can be set
#' (which correspond to arguments in that function).
#'
#' Note that the following fields are allowed in the construction of an \code{xgb.DMatrix}
#' but \bold{aren't} allowed here:\itemize{
#' \item data
#' \item missing
#' \item silent
#' \item nthread
#' }
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#'
#' labels <- getinfo(dtrain, 'label')
#' setinfo(dtrain, 'label', 1-labels)
#' labels2 <- getinfo(dtrain, 'label')
#' stopifnot(all.equal(labels2, 1-labels))
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

#' @title Get Quantile Cuts from DMatrix
#' @description Get the quantile cuts (a.k.a. borders) from an `xgb.DMatrix`
#' that has been quantized for the histogram method (`tree_method="hist"`).
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
#' @param dmat An `xgb.DMatrix` object, as returned by \link{xgb.DMatrix}.
#' @param output Output format for the quantile cuts. Possible options are:\itemize{
#' \item `"list"` will return the output as a list with one entry per column, where
#' each column will have a numeric vector with the cuts. The list will be named if
#' `dmat` has column names assigned to it.
#' \item `"arrays"` will return a list with entries `indptr` (base-0 indexing) and
#' `data`. Here, the cuts for column 'i' are obtained by slicing 'data' from entries
#' `indptr[i]+1` to `indptr[i+1]`.
#' }
#' @return The quantile cuts, in the format specified by parameter `output`.
#' @examples
#' library(xgboost)
#' data(mtcars)
#' y <- mtcars$mpg
#' x <- as.matrix(mtcars[, -1])
#' dm <- xgb.DMatrix(x, label = y, nthread = 1)
#'
#' # DMatrix is not quantized right away, but will be once a hist model is generated
#' model <- xgb.train(
#'   data = dm,
#'   params = list(
#'     tree_method = "hist",
#'     max_bin = 8,
#'     nthread = 1
#'   ),
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

#' @title Get Number of Non-Missing Entries in DMatrix
#' @param dmat An `xgb.DMatrix` object, as returned by \link{xgb.DMatrix}.
#' @return The number of non-missing entries in the DMatrix
#' @export
xgb.get.DMatrix.num.non.missing <- function(dmat) { # nolint
  stopifnot(inherits(dmat, "xgb.DMatrix"))
  return(.Call(XGDMatrixNumNonMissing_R, dmat))
}

#' @title Get DMatrix Data
#' @param dmat An `xgb.DMatrix` object, as returned by \link{xgb.DMatrix}.
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

#' Get a new DMatrix containing the specified rows of
#' original xgb.DMatrix object
#'
#' Get a new DMatrix containing the specified rows of
#' original xgb.DMatrix object
#'
#' @param object Object of class "xgb.DMatrix"
#' @param idxset a integer vector of indices of rows needed
#' @param colset currently not used (columns subsetting is not available)
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#'
#' dsub <- slice(dtrain, 1:42)
#' labels1 <- getinfo(dsub, 'label')
#' dsub <- dtrain[1:42, ]
#' labels2 <- getinfo(dsub, 'label')
#' all.equal(labels1, labels2)
#'
#' @rdname slice.xgb.DMatrix
#' @export
slice <- function(object, idxset) UseMethod("slice")

#' @rdname slice.xgb.DMatrix
#' @export
slice.xgb.DMatrix <- function(object, idxset) {
  if (!inherits(object, "xgb.DMatrix")) {
    stop("object must be xgb.DMatrix")
  }
  ret <- .Call(XGDMatrixSliceDMatrix_R, object, idxset)

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
  return(structure(ret, class = "xgb.DMatrix"))
}

#' @rdname slice.xgb.DMatrix
#' @export
`[.xgb.DMatrix` <- function(object, idxset, colset = NULL) {
  slice(object, idxset)
}


#' Print xgb.DMatrix
#'
#' Print information about xgb.DMatrix.
#' Currently it displays dimensions and presence of info-fields and colnames.
#'
#' @param x an xgb.DMatrix object
#' @param verbose whether to print colnames (when present)
#' @param ... not currently used
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' dtrain <- with(agaricus.train, xgb.DMatrix(data, label = label, nthread = 2))
#'
#' dtrain
#' print(dtrain, verbose=TRUE)
#'
#' @method print xgb.DMatrix
#' @export
print.xgb.DMatrix <- function(x, verbose = FALSE, ...) {
  if (.Call(XGCheckNullPtr_R, x)) {
    cat("INVALID xgb.DMatrix object. Must be constructed anew.\n")
    return(invisible(x))
  }
  cat('xgb.DMatrix  dim:', nrow(x), 'x', ncol(x), ' info: ')
  infos <- character(0)
  if (xgb.DMatrix.hasinfo(x, 'label')) infos <- 'label'
  if (xgb.DMatrix.hasinfo(x, 'weight')) infos <- c(infos, 'weight')
  if (xgb.DMatrix.hasinfo(x, 'base_margin')) infos <- c(infos, 'base_margin')
  if (length(infos) == 0) infos <- 'NA'
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
