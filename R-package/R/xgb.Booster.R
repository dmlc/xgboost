# Construct a Booster from cachelist
# internal utility function
xgb.Booster <- function(params = list(), cachelist = list(), modelfile = NULL) {
  if (typeof(cachelist) != "list") {
    stop("xgb.Booster: only accepts list of DMatrix as cachelist")
  }
  for (dm in cachelist) {
    if (class(dm) != "xgb.DMatrix") {
      stop("xgb.Booster: only accepts list of DMatrix as cachelist")
    }
  }
  handle <- .Call("XGBoosterCreate_R", cachelist, PACKAGE = "xgboost")
  if (length(params) != 0) {
    for (i in 1:length(params)) {
      p <- params[i]
      .Call("XGBoosterSetParam_R", handle, gsub("\\.", "_", names(p)), as.character(p),
            PACKAGE = "xgboost")
    }
  }
  if (!is.null(modelfile)) {
    if (typeof(modelfile) == "character") {
      .Call("XGBoosterLoadModel_R", handle, modelfile, PACKAGE = "xgboost")
    } else if (typeof(modelfile) == "raw") {
      .Call("XGBoosterLoadModelFromRaw_R", handle, modelfile, PACKAGE = "xgboost")
    } else {
      stop("xgb.Booster: modelfile must be character or raw vector")
    }
  }
  return(structure(handle, class = "xgb.Booster.handle"))
}

# Convert xgb.Booster.handle to xgb.Booster
# internal utility function
xgb.handleToBooster <- function(handle, raw = NULL)
{
  bst <- list(handle = handle, raw = raw)
  class(bst) <- "xgb.Booster"
  return(bst)
}

# Check whether an xgb.Booster object is complete
# internal utility function
xgb.Booster.check <- function(bst, saveraw = TRUE)
{
  isnull <- is.null(bst$handle)
  if (!isnull) {
    isnull <- .Call("XGCheckNullPtr_R", bst$handle, PACKAGE="xgboost")
  }
  if (isnull) {
    bst$handle <- xgb.Booster(modelfile = bst$raw)
  } else {
    if (is.null(bst$raw) && saveraw)
      bst$raw <- xgb.save.raw(bst$handle)
  }
  return(bst)
}

#' Predict method for eXtreme Gradient Boosting model
#' 
#' Predicted values based on either xgboost model or model handle object.
#' 
#' @param object Object of class \code{xgb.Booster} or \code{xgb.Booster.handle}
#' @param newdata takes \code{matrix}, \code{dgCMatrix}, local data file or 
#'   \code{xgb.DMatrix}. 
#' @param missing Missing is only used when input is dense matrix, pick a float 
#'     value that represents missing value. Sometime a data use 0 or other extreme value to represents missing values.
#' @param outputmargin whether the prediction should be shown in the original
#'   value of sum of functions, when outputmargin=TRUE, the prediction is 
#'   untransformed margin value. In logistic regression, outputmargin=T will
#'   output value before logistic transformation.
#' @param ntreelimit limit number of trees used in prediction, this parameter is
#'  only valid for gbtree, but not for gblinear. set it to be value bigger 
#'  than 0. It will use all trees by default.
#' @param predleaf whether predict leaf index instead. If set to TRUE, the output will be a matrix object.
#' @param ... Parameters pass to \code{predict.xgb.Booster}
#' 
#' @details  
#' The option \code{ntreelimit} purpose is to let the user train a model with lots 
#' of trees but use only the first trees for prediction to avoid overfitting 
#' (without having to train a new model with less trees).
#' 
#' The option \code{predleaf} purpose is inspired from §3.1 of the paper 
#' \code{Practical Lessons from Predicting Clicks on Ads at Facebook}.
#' The idea is to use the model as a generator of new features which capture non linear link 
#' from original features.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' 
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")
#' pred <- predict(bst, test$data)
#' @rdname predict.xgb.Booster
#' @export
predict.xgb.Booster <- function(object, newdata, missing = NA,
    outputmargin = FALSE, ntreelimit = NULL, predleaf = FALSE) {
  if (class(object) != "xgb.Booster"){
    stop("predict: model in prediction must be of class xgb.Booster")
  } else {
    object <- xgb.Booster.check(object, saveraw = FALSE)
  }
  if (class(newdata) != "xgb.DMatrix") {
    newdata <- xgb.DMatrix(newdata, missing = missing)
  }
  if (is.null(ntreelimit)) {
    ntreelimit <- 0
  } else {
    if (ntreelimit < 1){
      stop("predict: ntreelimit must be equal to or greater than 1")
    }
  }
  option <- 0
  if (outputmargin) {
    option <- option + 1
  }
  if (predleaf) {
    option <- option + 2
  }
  ret <- .Call("XGBoosterPredict_R", object$handle, newdata, as.integer(option),
               as.integer(ntreelimit), PACKAGE = "xgboost")
  if (predleaf){
      len <- getinfo(newdata, "nrow")
      if (length(ret) == len){
          ret <- matrix(ret,ncol = 1)
      } else {
          ret <- matrix(ret, ncol = len)
          ret <- t(ret)
      }
  }
  return(ret)
}

#' @rdname predict.xgb.Booster
#' @export
predict.xgb.Booster.handle <- function(object, ...) {

  bst <- xgb.handleToBooster(object)

  ret <- predict(bst, ...)
  return(ret)
}
