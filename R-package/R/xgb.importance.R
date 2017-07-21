#' Importance of features in a model.
#' 
#' Creates a \code{data.table} of feature importances in a model.
#' 
#' @param feature_names character vector of feature names. If the model already
#'       contains feature names, those would be used when \code{feature_names=NULL} (default value).
#'       Non-null \code{feature_names} could be provided to override those in the model.
#' @param model object of class \code{xgb.Booster}.
#' @param trees (only for the gbtree booster) an integer vector of tree indices that should be included
#'          into the importance calculation. If set to \code{NULL}, all trees of the model are parsed.
#'          It could be useful, e.g., in multiclass classification to get feature importances 
#'          for each class separately. IMPORTANT: the tree index in xgboost models
#'          is zero-based (e.g., use \code{trees = 0:4} for first 5 trees).
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
#' A linear model's importance \code{data.table} has the following columns:
#' \itemize{
#'   \item \code{Features} names of the features used in the model;
#'   \item \code{Weight} the linear coefficient of this feature;
#'   \item \code{Class} (only for multiclass models) class label.
#' }
#' 
#' If \code{feature_names} is not provided and \code{model} doesn't have \code{feature_names}, 
#' index of the features will be used instead. Because the index is extracted from the model dump
#' (based on C++ code), it starts at 0 (as in C/C++ or Python) instead of 1 (usual in R).
#' 
#' @examples
#' 
#' # binomial classification using gbtree:
#' data(agaricus.train, package='xgboost')
#' bst <- xgboost(data = agaricus.train$data, label = agaricus.train$label, max_depth = 2, 
#'                eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
#' xgb.importance(model = bst)
#' 
#' # binomial classification using gblinear:
#' bst <- xgboost(data = agaricus.train$data, label = agaricus.train$label, booster = "gblinear", 
#'                eta = 0.3, nthread = 1, nrounds = 20, objective = "binary:logistic")
#' xgb.importance(model = bst)
#' 
#' # multiclass classification using gbtree:
#' nclass <- 3
#' nrounds <- 10
#' mbst <- xgboost(data = as.matrix(iris[, -5]), label = as.numeric(iris$Species) - 1,
#'                max_depth = 3, eta = 0.2, nthread = 2, nrounds = nrounds,
#'                objective = "multi:softprob", num_class = nclass)
#' # all classes clumped together:
#' xgb.importance(model = mbst)
#' # inspect importances separately for each class:
#' xgb.importance(model = mbst, trees = seq(from=0, by=nclass, length.out=nrounds))
#' xgb.importance(model = mbst, trees = seq(from=1, by=nclass, length.out=nrounds))
#' xgb.importance(model = mbst, trees = seq(from=2, by=nclass, length.out=nrounds))
#' 
#' # multiclass classification using gblinear:
#' mbst <- xgboost(data = scale(as.matrix(iris[, -5])), label = as.numeric(iris$Species) - 1,
#'                booster = "gblinear", eta = 0.2, nthread = 1, nrounds = 15,
#'                objective = "multi:softprob", num_class = nclass)
#' xgb.importance(model = mbst)
#'
#' @export
xgb.importance <- function(feature_names = NULL, model = NULL, trees = NULL,
                           data = NULL, label = NULL, target = NULL){
  
  if (!(is.null(data) && is.null(label) && is.null(target)))
    warning("xgb.importance: parameters 'data', 'label' and 'target' are deprecated")
  
  if (!inherits(model, "xgb.Booster"))
    stop("model: must be an object of class xgb.Booster")
  
  if (is.null(feature_names) && !is.null(model$feature_names))
    feature_names <- model$feature_names
  
  if (!(is.null(feature_names) || is.character(feature_names)))
    stop("feature_names: Has to be a character vector")

  model_text_dump <- xgb.dump(model = model, with_stats = TRUE)
  
  # linear model
  if(model_text_dump[2] == "bias:"){
    weights <- which(model_text_dump == "weight:") %>%
               {model_text_dump[(. + 1):length(model_text_dump)]} %>%
               as.numeric
    
    num_class <- NVL(model$params$num_class, 1)
    if(is.null(feature_names)) 
      feature_names <- seq(to = length(weights) / num_class) - 1
    if (length(feature_names) * num_class != length(weights))
      stop("feature_names length does not match the number of features used in the model")
    
    result <- if (num_class == 1) {
      data.table(Feature = feature_names, Weight = weights)[order(-abs(Weight))]
    } else {
      data.table(Feature = rep(feature_names, each = num_class),
                 Weight = weights,
                 Class = seq_len(num_class) - 1)[order(Class, -abs(Weight))]
    }
  } else { 
  # tree model
    result <- xgb.model.dt.tree(feature_names = feature_names,
                                text = model_text_dump,
                                trees = trees)[
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
