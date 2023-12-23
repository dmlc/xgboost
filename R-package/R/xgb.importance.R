#' Feature importance
#'
#' Creates a `data.table` of feature importances.
#'
#' @param feature_names Character vector used to overwrite the feature names
#'        of the model. The default is `NULL` (use original feature names).
#' @param model Object of class `xgb.Booster`.
#' @param trees An integer vector of tree indices that should be included
#'        into the importance calculation (only for the "gbtree" booster).
#'        The default (`NULL`) parses all trees.
#'        It could be useful, e.g., in multiclass classification to get feature importances
#'        for each class separately. *Important*: the tree index in XGBoost models
#'        is zero-based (e.g., use `trees = 0:4` for the first five trees).
#' @param data Deprecated.
#' @param label Deprecated.
#' @param target Deprecated.
#'
#' @details
#'
#' This function works for both linear and tree models.
#'
#' For linear models, the importance is the absolute magnitude of linear coefficients.
#' To obtain a meaningful ranking by importance for linear models, the features need to
#' be on the same scale (which is also recommended when using L1 or L2 regularization).
#'
#' @return A `data.table` with the following columns:
#'
#' For a tree model:
#' - `Features`: Names of the features used in the model.
#' - `Gain`: Fractional contribution of each feature to the model based on
#'    the total gain of this feature's splits. Higher percentage means higher importance.
#' - `Cover`: Metric of the number of observation related to this feature.
#' - `Frequency`: Percentage of times a feature has been used in trees.
#'
#' For a linear model:
#' - `Features`: Names of the features used in the model.
#' - `Weight`: Linear coefficient of this feature.
#' - `Class`: Class label (only for multiclass models).
#'
#' If `feature_names` is not provided and `model` doesn't have `feature_names`,
#' the index of the features will be used instead. Because the index is extracted from the model dump
#' (based on C++ code), it starts at 0 (as in C/C++ or Python) instead of 1 (usual in R).
#'
#' @examples
#'
#' # binomial classification using "gbtree":
#' data(agaricus.train, package = "xgboost")
#'
#' bst <- xgboost(
#'   data = agaricus.train$data,
#'   label = agaricus.train$label,
#'   max_depth = 2,
#'   eta = 1,
#'   nthread = 2,
#'   nrounds = 2,
#'   objective = "binary:logistic"
#' )
#'
#' xgb.importance(model = bst)
#'
#' # binomial classification using "gblinear":
#' bst <- xgboost(
#'   data = agaricus.train$data,
#'   label = agaricus.train$label,
#'   booster = "gblinear",
#'   eta = 0.3,
#'   nthread = 1,
#'   nrounds = 20,objective = "binary:logistic"
#' )
#'
#' xgb.importance(model = bst)
#'
#' # multiclass classification using "gbtree":
#' nclass <- 3
#' nrounds <- 10
#' mbst <- xgboost(
#'   data = as.matrix(iris[, -5]),
#'   label = as.numeric(iris$Species) - 1,
#'   max_depth = 3,
#'   eta = 0.2,
#'   nthread = 2,
#'   nrounds = nrounds,
#'   objective = "multi:softprob",
#'   num_class = nclass
#' )
#'
#' # all classes clumped together:
#' xgb.importance(model = mbst)
#'
#' # inspect importances separately for each class:
#' xgb.importance(
#'   model = mbst, trees = seq(from = 0, by = nclass, length.out = nrounds)
#' )
#' xgb.importance(
#'   model = mbst, trees = seq(from = 1, by = nclass, length.out = nrounds)
#' )
#' xgb.importance(
#'   model = mbst, trees = seq(from = 2, by = nclass, length.out = nrounds)
#' )
#'
#' # multiclass classification using "gblinear":
#' mbst <- xgboost(
#'   data = scale(as.matrix(iris[, -5])),
#'   label = as.numeric(iris$Species) - 1,
#'   booster = "gblinear",
#'   eta = 0.2,
#'   nthread = 1,
#'   nrounds = 15,
#'   objective = "multi:softprob",
#'   num_class = nclass
#' )
#'
#' xgb.importance(model = mbst)
#'
#' @export
xgb.importance <- function(feature_names = NULL, model = NULL, trees = NULL,
                           data = NULL, label = NULL, target = NULL) {

  if (!(is.null(data) && is.null(label) && is.null(target)))
    warning("xgb.importance: parameters 'data', 'label' and 'target' are deprecated")

  if (!inherits(model, "xgb.Booster"))
    stop("model: must be an object of class xgb.Booster")

  if (is.null(feature_names) && !is.null(model$feature_names))
    feature_names <- model$feature_names

  if (!(is.null(feature_names) || is.character(feature_names)))
    stop("feature_names: Has to be a character vector")

  model <- xgb.Booster.complete(model)
  config <- jsonlite::fromJSON(xgb.config(model))
  if (config$learner$gradient_booster$name == "gblinear") {
    args <- list(importance_type = "weight", feature_names = feature_names)
    results <- .Call(
      XGBoosterFeatureScore_R, model$handle, jsonlite::toJSON(args, auto_unbox = TRUE, null = "null")
    )
    names(results) <- c("features", "shape", "weight")
    if (length(results$shape) == 2) {
        n_classes <- results$shape[2]
    } else {
        n_classes <- 0
    }
    importance <- if (n_classes == 0) {
      data.table(Feature = results$features, Weight = results$weight)[order(-abs(Weight))]
    } else {
      data.table(
        Feature = rep(results$features, each = n_classes), Weight = results$weight, Class = seq_len(n_classes) - 1
      )[order(Class, -abs(Weight))]
    }
  } else {
    concatenated <- list()
    output_names <- vector()
    for (importance_type in c("weight", "total_gain", "total_cover")) {
      args <- list(importance_type = importance_type, feature_names = feature_names, tree_idx = trees)
      results <- .Call(
        XGBoosterFeatureScore_R, model$handle, jsonlite::toJSON(args, auto_unbox = TRUE, null = "null")
      )
      names(results) <- c("features", "shape", importance_type)
      concatenated[
        switch(importance_type, "weight" = "Frequency", "total_gain" = "Gain", "total_cover" = "Cover")
      ] <- results[importance_type]
      output_names <- results$features
    }
    importance <- data.table(
        Feature = output_names,
        Gain = concatenated$Gain / sum(concatenated$Gain),
        Cover = concatenated$Cover / sum(concatenated$Cover),
        Frequency = concatenated$Frequency / sum(concatenated$Frequency)
    )[order(Gain, decreasing = TRUE)]
  }
  importance
}

# Avoid error messages during CRAN check.
# The reason is that these variables are never declared
# They are mainly column names inferred by Data.table...
globalVariables(c(".", ".N", "Gain", "Cover", "Frequency", "Feature", "Class"))
