#' Dump an xgboost model in text format.
#' 
#' Dump an xgboost model in text format.
#' 
#' @param model the model object.
#' @param fname the name of the text file where to save the model text dump. 
#'        If not provided or set to \code{NULL}, the model is returned as a \code{character} vector.
#' @param fmap feature map file representing feature types.
#'        Detailed description could be found at 
#'        \url{https://github.com/dmlc/xgboost/wiki/Binary-Classification#dump-model}.
#'        See demo/ for walkthrough example in R, and
#'        \url{https://github.com/dmlc/xgboost/blob/master/demo/data/featmap.txt} 
#'        for example Format.
#' @param with_stats whether to dump some additional statistics about the splits.
#'        When this option is on, the model dump contains two additional values:
#'        gain is the approximate loss function gain we get in each split;
#'        cover is the sum of second order gradient in each node.
#' @param dump_format either 'text' or 'json' format could be specified.
#' @param ... currently not used
#'
#' @return
#' If fname is not provided or set to \code{NULL} the function will return the model
#' as a \code{character} vector. Otherwise it will return \code{TRUE}.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max_depth = 2, 
#'                eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
#' # save the model in file 'xgb.model.dump'
#' xgb.dump(bst, 'xgb.model.dump', with_stats = TRUE)
#' 
#' # print the model without saving it to a file
#' print(xgb.dump(bst, with_stats = TRUE))
#' 
#' # print in JSON format:
#' cat(xgb.dump(bst, with_stats = TRUE, dump_format='json'))
#' 
#' @export
xgb.dump <- function(model, fname = NULL, fmap = "", with_stats=FALSE,
                     dump_format = c("text", "json"), ...) {
  check.deprecation(...)
  dump_format <- match.arg(dump_format)
  if (!inherits(model, "xgb.Booster"))
    stop("model: argument must be of type xgb.Booster")
  if (!(is.null(fname) || is.character(fname)))
    stop("fname: argument must be a character string (when provided)")
  if (!(is.null(fmap) || is.character(fmap)))
    stop("fmap: argument must be a character string (when provided)")
  
  model <- xgb.Booster.complete(model)
  model_dump <- .Call(XGBoosterDumpModel_R, model$handle, NVL(fmap, "")[1], as.integer(with_stats),
                      as.character(dump_format))

  if (is.null(fname)) 
    model_dump <- stri_replace_all_regex(model_dump, '\t', '')
  
  if (dump_format == "text")
    model_dump <- unlist(stri_split_regex(model_dump, '\n'))
  
  model_dump <- grep('^\\s*$', model_dump, invert = TRUE, value = TRUE)
  
  if (is.null(fname)) {
    return(model_dump)
  } else {
    writeLines(model_dump, fname[1])
    return(TRUE)
  }
}
