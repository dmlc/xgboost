#' Save xgboost model to text file
#' 
#' Save a xgboost model to text file. Could be parsed later.
#' 
#' @importFrom magrittr %>%
#' @importFrom stringr str_replace
#' @importFrom data.table fread
#' @importFrom data.table :=
#' @importFrom data.table setnames
#' @param model the model object.
#' @param fname the name of the text file where to save the model text dump. If not provided or set to \code{NULL} the function will return the model as a \code{character} vector.
#' @param fmap feature map file representing the type of feature. 
#'        Detailed description could be found at 
#'        \url{https://github.com/dmlc/xgboost/wiki/Binary-Classification#dump-model}.
#'        See demo/ for walkthrough example in R, and
#'        \url{https://github.com/dmlc/xgboost/blob/master/demo/data/featmap.txt} 
#'        for example Format.
#' @param with.stats whether dump statistics of splits 
#'        When this option is on, the model dump comes with two additional statistics:
#'        gain is the approximate loss function gain we get in each split;
#'        cover is the sum of second order gradient in each node.
#'
#' @return
#' if fname is not provided or set to \code{NULL} the function will return the model as a \code{character} vector. Otherwise it will return \code{TRUE}.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")
#' # save the model in file 'xgb.model.dump'
#' xgb.dump(bst, 'xgb.model.dump', with.stats = TRUE)
#' 
#' # print the model without saving it to a file
#' print(xgb.dump(bst))
#' @export
#' 
xgb.dump <- function(model = NULL, fname = NULL, fmap = "", with.stats=FALSE) {
  if (class(model) != "xgb.Booster") {
    stop("model: argument must be type xgb.Booster")
  } else {
    model <- xgb.Booster.check(model)
  }
  if (!(class(fname) %in% c("character", "NULL") && length(fname) <= 1)) {
    stop("fname: argument must be type character (when provided)")
  }
  if (!(class(fmap) %in% c("character", "NULL") && length(fname) <= 1)) {
    stop("fmap: argument must be type character (when provided)")
  }
  
  longString <- .Call("XGBoosterDumpModel_R", model$handle, fmap, as.integer(with.stats), PACKAGE = "xgboost")
  
  dt <- fread(paste(longString, collapse = ""), sep = "\n", header = F)

  setnames(dt, "Lines")
  
  if(is.null(fname)) {
    result <- dt[Lines != "0"][, Lines := str_replace(Lines, "^\t+", "")][Lines != ""][, paste(Lines)]
    return(result)
  } else {
    result <- dt[Lines != "0"][Lines != ""][, paste(Lines)] %>% writeLines(fname)
    return(TRUE)
  }
}

# Avoid error messages during CRAN check.
# The reason is that these variables are never declared
# They are mainly column names inferred by Data.table...
globalVariables(c("Lines", "."))
