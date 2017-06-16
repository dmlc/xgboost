#' Construct xgb.DMatrix object
#' 
#' Construct xgb.DMatrix object from either a dense matrix, a sparse matrix, or a local file.
#' Supported input file formats are either a libsvm text file or a binary file that was created previously by
#' \code{\link{xgb.DMatrix.save}}).
#' 
#' @param data a \code{matrix} object (either numeric or integer), a \code{dgCMatrix} object, or a character 
#'        string representing a filename.
#' @param info a named list of additional information to store in the \code{xgb.DMatrix} object.
#'        See \code{\link{setinfo}} for the specific allowed kinds of 
#' @param missing a float value to represents missing values in data (used only when input is a dense matrix).
#'        It is useful when a 0 or some other extreme value represents missing values in data.
#' @param silent whether to suppress printing an informational message after loading from a file.
#' @param ... the \code{info} data could be passed directly as parameters, without creating an \code{info} list.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' xgb.DMatrix.save(dtrain, 'xgb.DMatrix.data')
#' dtrain <- xgb.DMatrix('xgb.DMatrix.data')
#' @export
xgb.DMatrix <- function(data, info = list(), missing = NA, silent = FALSE, ...) {
  cnames <- NULL
  if (typeof(data) == "character") {
    if (length(data) > 1)
      stop("'data' has class 'character' and length ", length(data),
           ".\n  'data' accepts either a numeric matrix or a single filename.")
    handle <- .Call(XGDMatrixCreateFromFile_R, data, as.integer(silent))
  } else if (is.matrix(data)) {
    handle <- .Call(XGDMatrixCreateFromMat_R, data, missing)
    cnames <- colnames(data)
  } else if (inherits(data, "dgCMatrix")) {
    handle <- .Call(XGDMatrixCreateFromCSC_R, data@p, data@i, data@x, nrow(data))
    cnames <- colnames(data)
  } else {
    stop("xgb.DMatrix does not support construction from ", typeof(data))
  }
  dmat <- handle
  attributes(dmat) <- list(.Dimnames = list(NULL, cnames), class = "xgb.DMatrix")

  info <- append(info, list(...))
  for (i in seq_along(info)) {
    p <- info[i]
    setinfo(dmat, names(p), p[[1]])
  }
  return(dmat)
}


# get dmatrix from data, label
# internal helper method
xgb.get.DMatrix <- function(data, label = NULL, missing = NA, weight = NULL) {
  if (inherits(data, "dgCMatrix") || is.matrix(data)) {
    if (is.null(label)) {
      stop("label must be provided when data is a matrix")
    }
    dtrain <- xgb.DMatrix(data, label = label, missing = missing)
    if (!is.null(weight)){
      setinfo(dtrain, "weight", weight)
    }
  } else {
    if (!is.null(label)) {
      warning("xgboost: label will be ignored.")
    }
    if (is.character(data)) {
      dtrain <- xgb.DMatrix(data[1])
    } else if (inherits(data, "xgb.DMatrix")) {
      dtrain <- data
    } else if (inherits(data, "data.frame")) {
      stop("xgboost doesn't support data.frame as input. Convert it to matrix first.")
    } else {
      stop("xgboost: invalid input data")
    }
  }
  return (dtrain)
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
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
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
#' row names would have no effect and returnten row names would be NULL.
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
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' dimnames(dtrain)
#' colnames(dtrain)
#' colnames(dtrain) <- make.names(1:ncol(train$data))
#' print(dtrain, verbose=TRUE)
#' 
#' @rdname dimnames.xgb.DMatrix
#' @export
dimnames.xgb.DMatrix <- function(x) {
  attr(x, '.Dimnames')
}

#' @rdname dimnames.xgb.DMatrix
#' @export
`dimnames<-.xgb.DMatrix` <- function(x, value) {
  if (!is.list(value) || length(value) != 2L)
    stop("invalid 'dimnames' given: must be a list of two elements")
  if (!is.null(value[[1L]]))
    stop("xgb.DMatrix does not have rownames")
  if (is.null(value[[2]])) {
    attr(x, '.Dimnames') <- NULL
    return(x)
  }
  if (ncol(x) != length(value[[2]])) 
    stop("can't assign ", length(value[[2]]), " colnames to a ", 
         ncol(x), " column xgb.DMatrix")
  attr(x, '.Dimnames') <- value
  x
}


#' Get information of an xgb.DMatrix object
#' 
#' Get information of an xgb.DMatrix object
#' @param object Object of class \code{xgb.DMatrix}
#' @param name the name of the information field to get (see details)
#' @param ... other parameters
#' 
#' @details
#' The \code{name} field can be one of the following:
#' 
#' \itemize{
#'     \item \code{label}: label Xgboost learn from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item \code{base_margin}: base margin is the base prediction Xgboost will boost from ;
#'     \item \code{nrow}: number of rows of the \code{xgb.DMatrix}.
#'     
#' }
#' 
#' \code{group} can be setup by \code{setinfo} but can't be retrieved by \code{getinfo}.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' 
#' labels <- getinfo(dtrain, 'label')
#' setinfo(dtrain, 'label', 1-labels)
#' 
#' labels2 <- getinfo(dtrain, 'label')
#' stopifnot(all(labels2 == 1-labels))
#' @rdname getinfo
#' @export
getinfo <- function(object, ...) UseMethod("getinfo")

#' @rdname getinfo
#' @export
getinfo.xgb.DMatrix <- function(object, name, ...) {
  if (typeof(name) != "character" ||
      length(name) != 1 ||
      !name %in% c('label', 'weight', 'base_margin', 'nrow')) {
    stop("getinfo: name must be one of the following\n",
         "    'label', 'weight', 'base_margin', 'nrow'")
  }
  if (name != "nrow"){
    ret <- .Call(XGDMatrixGetInfo_R, object, name)
  } else {
    ret <- nrow(object)
  }
  if (length(ret) == 0) return(NULL)
  return(ret)
}


#' Set information of an xgb.DMatrix object
#' 
#' Set information of an xgb.DMatrix object
#' 
#' @param object Object of class "xgb.DMatrix"
#' @param name the name of the field to get
#' @param info the specific field of information to set
#' @param ... other parameters
#'
#' @details
#' The \code{name} field can be one of the following:
#' 
#' \itemize{
#'     \item \code{label}: label Xgboost learn from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item \code{base_margin}: base margin is the base prediction Xgboost will boost from ;
#'     \item \code{group}: number of rows in each group (to use with \code{rank:pairwise} objective).
#' }
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' 
#' labels <- getinfo(dtrain, 'label')
#' setinfo(dtrain, 'label', 1-labels)
#' labels2 <- getinfo(dtrain, 'label')
#' stopifnot(all.equal(labels2, 1-labels))
#' @rdname setinfo
#' @export
setinfo <- function(object, ...) UseMethod("setinfo")

#' @rdname setinfo
#' @export
setinfo.xgb.DMatrix <- function(object, name, info, ...) {
  if (name == "label") {
    if (length(info) != nrow(object))
      stop("The length of labels must equal to the number of rows in the input data")
    .Call(XGDMatrixSetInfo_R, object, name, as.numeric(info))
    return(TRUE)
  }
  if (name == "weight") {
    if (length(info) != nrow(object))
      stop("The length of weights must equal to the number of rows in the input data")
    .Call(XGDMatrixSetInfo_R, object, name, as.numeric(info))
    return(TRUE)
  }
  if (name == "base_margin") {
    # if (length(info)!=nrow(object))
    #   stop("The length of base margin must equal to the number of rows in the input data")
    .Call(XGDMatrixSetInfo_R, object, name, as.numeric(info))
    return(TRUE)
  }
  if (name == "group") {
    if (sum(info) != nrow(object))
      stop("The sum of groups must equal to the number of rows in the input data")
    .Call(XGDMatrixSetInfo_R, object, name, as.integer(info))
    return(TRUE)
  }
  stop("setinfo: unknown info name ", name)
  return(FALSE)
}


#' Get a new DMatrix containing the specified rows of
#' orginal xgb.DMatrix object
#'
#' Get a new DMatrix containing the specified rows of
#' orginal xgb.DMatrix object
#' 
#' @param object Object of class "xgb.DMatrix"
#' @param idxset a integer vector of indices of rows needed
#' @param colset currently not used (columns subsetting is not available)
#' @param ... other parameters (currently not used)
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' 
#' dsub <- slice(dtrain, 1:42)
#' labels1 <- getinfo(dsub, 'label')
#' dsub <- dtrain[1:42, ]
#' labels2 <- getinfo(dsub, 'label')
#' all.equal(labels1, labels2)
#' 
#' @rdname slice.xgb.DMatrix
#' @export
slice <- function(object, ...) UseMethod("slice")

#' @rdname slice.xgb.DMatrix
#' @export
slice.xgb.DMatrix <- function(object, idxset, ...) {
  if (!inherits(object, "xgb.DMatrix")) {
    stop("object must be xgb.DMatrix")
  }
  ret <- .Call(XGDMatrixSliceDMatrix_R, object, idxset)

  attr_list <- attributes(object)
  nr <- nrow(object)
  len <- sapply(attr_list, length)
  ind <- which(len == nr)
  if (length(ind) > 0) {
    nms <- names(attr_list)[ind]
    for (i in seq_along(ind)) {
      attr(ret, nms[i]) <- attr(object, nms[i])[idxset]
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
#' train <- agaricus.train
#' dtrain <- xgb.DMatrix(train$data, label=train$label)
#' 
#' dtrain
#' print(dtrain, verbose=TRUE)
#' 
#' @method print xgb.DMatrix
#' @export
print.xgb.DMatrix <- function(x, verbose = FALSE, ...) {
  cat('xgb.DMatrix  dim:', nrow(x), 'x', ncol(x), ' info: ')
  infos <- c()
  if(length(getinfo(x, 'label')) > 0) infos <- 'label'
  if(length(getinfo(x, 'weight')) > 0) infos <- c(infos, 'weight')
  if(length(getinfo(x, 'base_margin')) > 0) infos <- c(infos, 'base_margin')
  if (length(infos) == 0) infos <- 'NA'
  cat(infos)
  cnames <- colnames(x)
  cat('  colnames:')
  if (verbose & !is.null(cnames)) {
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
