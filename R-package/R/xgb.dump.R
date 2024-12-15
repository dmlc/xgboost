#' Dump an XGBoost model in text format.
#'
#' Dump an XGBoost model in text format.
#'
#' @param model The model object.
#' @param fname The name of the text file where to save the model text dump.
#'   If not provided or set to `NULL`, the model is returned as a character vector.
#' @param fmap Feature map file representing feature types. See demo/ for a walkthrough
#'   example in R, and \url{https://github.com/dmlc/xgboost/blob/master/demo/data/featmap.txt}
#'   to see an example of the value.
#' @param with_stats Whether to dump some additional statistics about the splits.
#'   When this option is on, the model dump contains two additional values:
#'   gain is the approximate loss function gain we get in each split;
#'   cover is the sum of second order gradient in each node.
#' @param dump_format Either 'text', 'json', or 'dot' (graphviz) format could be specified.
#'
#'   Format 'dot' for a single tree can be passed directly to packages that consume this format
#'   for graph visualization, such as function `DiagrammeR::grViz()`
#' @inheritParams xgb.train
#' @return
#' If fname is not provided or set to `NULL` the function will return the model
#' as a character vector. Otherwise it will return `TRUE`.
#'
#' @examples
#' \dontshow{RhpcBLASctl::omp_set_num_threads(1)}
#' data(agaricus.train, package = "xgboost")
#' data(agaricus.test, package = "xgboost")
#'
#' train <- agaricus.train
#' test <- agaricus.test
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(train$data, label = train$label),
#'   nrounds = 2,
#'   params = xgb.params(
#'     max_depth = 2,
#'     nthread = 2,
#'     objective = "binary:logistic"
#'   )
#' )
#'
#' # save the model in file 'xgb.model.dump'
#' dump_path = file.path(tempdir(), 'model.dump')
#' xgb.dump(bst, dump_path, with_stats = TRUE)
#'
#' # print the model without saving it to a file
#' print(xgb.dump(bst, with_stats = TRUE))
#'
#' # print in JSON format:
#' cat(xgb.dump(bst, with_stats = TRUE, dump_format = "json"))
#'
#' # plot first tree leveraging the 'dot' format
#' if (requireNamespace('DiagrammeR', quietly = TRUE)) {
#'   DiagrammeR::grViz(xgb.dump(bst, dump_format = "dot")[[1L]])
#' }
#' @export
xgb.dump <- function(model, fname = NULL, fmap = "", with_stats = FALSE,
                     dump_format = c("text", "json", "dot"), ...) {
  check.deprecation(deprecated_dump_params, match.call(), ...)
  dump_format <- match.arg(dump_format)
  if (!inherits(model, "xgb.Booster"))
    stop("model: argument must be of type xgb.Booster")
  if (!(is.null(fname) || is.character(fname)))
    stop("fname: argument must be a character string (when provided)")
  if (!(is.null(fmap) || is.character(fmap)))
    stop("fmap: argument must be a character string (when provided)")

  model_dump <- .Call(
    XGBoosterDumpModel_R,
    xgb.get.handle(model),
    NVL(fmap, "")[1],
    as.integer(with_stats),
    as.character(dump_format)
  )
  if (dump_format == "dot") {
    return(sapply(model_dump, function(x) gsub("^booster\\[\\d+\\]\\n", "\\1", x)))
  }

  if (is.null(fname))
    model_dump <- gsub('\t', '', model_dump, fixed = TRUE)

  if (dump_format == "text")
    model_dump <- unlist(strsplit(model_dump, '\n', fixed = TRUE))

  model_dump <- grep('^\\s*$', model_dump, invert = TRUE, value = TRUE)

  if (is.null(fname)) {
    return(model_dump)
  } else {
    fname <- path.expand(fname)
    writeLines(model_dump, fname[1])
    return(TRUE)
  }
}
