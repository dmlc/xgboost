#' Save xgboost model to text file
#' 
#' Save a xgboost model to text file. Could be parsed later.
#' 
#' @importFrom magrittr %>%
#' @importFrom stringr str_split
#' @importFrom stringr str_replace_all
#' @param model the model object.
#' @param fname the name of the text file where to save the model. If not provided or set to \code{NULL} the function will return the model as a \code{character} vector.
#' @param fmap feature map file representing the type of feature. 
#'        Detailed description could be found at 
#'        \url{https://github.com/tqchen/xgboost/wiki/Binary-Classification#dump-model}.
#'        See demo/ for walkthrough example in R, and
#'        \url{https://github.com/tqchen/xgboost/blob/master/demo/data/featmap.txt} 
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
#'                eta = 1, nround = 2,objective = "binary:logistic")
#' xgb.dump(bst, 'xgb.model.dump')
#' @export
#' 
xgb.dump <- function(model, fname = NULL, fmap = "", with.stats=FALSE) {
  if (class(model) != "xgb.Booster") {
    stop("xgb.dump: first argument must be type xgb.Booster")
  }
  if (!class(fname) %in% c("character", "NULL")) {
    stop("xgb.dump: second argument must be type character if provided")
  }
  result <- .Call("XGBoosterDumpModel_R", model, fmap, as.integer(with.stats), PACKAGE = "xgboost")
  
  if(is.null(fname)) return(str_split(result, "\n") %>% unlist %>% str_replace_all("\t","  ") %>% Filter(function(x) x != "", .))
  
  writeLines(result, fname)
  TRUE
} 
