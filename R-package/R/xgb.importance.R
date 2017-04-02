#' Importance of features in a model.
#' 
#' Creates a \code{data.table} of feature importances in a model.
#' 
#' @param feature_names character vector of feature names. If the model already
#'       contains feature names, those would be used when \code{feature_names=NULL} (default value).
#'       Non-null \code{feature_names} could be provided to override those in the model.
#' @param model object of class \code{xgb.Booster}.
#' @param data deprecated.
#' @param label deprecated.
#' @param target deprecated.
#'
#' @details 
#' 
#' This function works for both linear and tree models.
#' 
#' For linear models, the importance is the absolute magnitude of linear coefficients. 
#' For that reason, in order to obtain a meaningful ranking by importance for a linear model, 
#' the features need to be on the same scale (which you also would want to do when using either 
#' L1 or L2 regularization).
#' 
#' @return
#' 
#' For a tree model, a \code{data.table} with the following columns:
#' \itemize{
#'   \item \code{Features} names of the features used in the model;
#'   \item \code{Gain} represents fractional contribution of each feature to the model based on
#'        the total gain of this feature's splits. Higher percentage means a more important 
#'        predictive feature.
#'   \item \code{Cover} metric of the number of observation related to this feature;
#'   \item \code{Frequency} percentage representing the relative number of times
#'        a feature have been used in trees.
#' }
#' 
#' A linear model's importance \code{data.table} has only two columns:
#' \itemize{
#'   \item \code{Features} names of the features used in the model;
#'   \item \code{Weight} the linear coefficient of this feature.
#' }
#' 
#' If you don't provide or \code{model} doesn't have \code{feature_names}, 
#' index of the features will be used instead. Because the index is extracted from the model dump
#' (based on C++ code), it starts at 0 (as in C/C++ or Python) instead of 1 (usual in R).
#' 
#' @examples
#' 
#' data(agaricus.train, package='xgboost')
#' 
#' bst <- xgboost(data = agaricus.train$data, label = agaricus.train$label, max_depth = 2, 
#'                eta = 1, nthread = 2, nrounds = 2,objective = "binary:logistic")
#' 
#' xgb.importance(model = bst)
#' 
#' @export
xgb.importance <- function(feature_names = NULL, model = NULL, 
                           data = NULL, label = NULL, target = NULL){
  
  if (!(is.null(data) && is.null(label) && is.null(target)))
    warning("xgb.importance: parameters 'data', 'label' and 'target' are deprecated")
  
  if (class(model) != "xgb.Booster")
    stop("Either 'model' has to be an object of class xgb.Booster")
  
  if (is.null(feature_names) && !is.null(model$feature_names))
    feature_names <- model$feature_names
  
  if (!class(feature_names) %in% c("character", "NULL"))
    stop("feature_names: Has to be a character vector")

  model_text_dump <- xgb.dump(model = model, with_stats = TRUE)
  
  # linear model
  if(model_text_dump[2] == "bias:"){
    weights <- which(model_text_dump == "weight:") %>%
               {model_text_dump[(. + 1):length(model_text_dump)]} %>%
               as.numeric
    if(is.null(feature_names)) 
      feature_names <- seq(to = length(weights))
    result <- data.table(Feature = feature_names, Weight = weights)[order(-abs(Weight))]
  } else { 
  # tree model
    result <- xgb.model.dt.tree(feature_names = feature_names, text = model_text_dump)[
      Feature != "Leaf", .(Gain = sum(Quality), 
                           Cover = sum(Cover), 
                           Frequency = .N), by = Feature][
      ,`:=`(Gain = Gain / sum(Gain), 
            Cover = Cover / sum(Cover),
            Frequency = Frequency / sum(Frequency))][
      order(Gain, decreasing = TRUE)]
  }
  result
}

# Avoid error messages during CRAN check.
# The reason is that these variables are never declared
# They are mainly column names inferred by Data.table...
globalVariables(c(".", ".N", "Gain", "Cover", "Frequency", "Feature"))
