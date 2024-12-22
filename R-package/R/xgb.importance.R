#' Feature importance
#'
#' Creates a `data.table` of feature importances.
#'
#' @details
#' This function works for both linear and tree models.
#'
#' For linear models, the importance is the absolute magnitude of linear coefficients.
#' To obtain a meaningful ranking by importance for linear models, the features need to
#' be on the same scale (which is also recommended when using L1 or L2 regularization).
#'
#' @param feature_names Character vector used to overwrite the feature names
#'   of the model. The default is `NULL` (use original feature names).
#' @param model Object of class `xgb.Booster`.
#' @param trees An integer vector of (base-1) tree indices that should be included
#'   into the importance calculation (only for the "gbtree" booster).
#'   The default (`NULL`) parses all trees.
#'   It could be useful, e.g., in multiclass classification to get feature importances
#'   for each class separately.
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
#' - `Class`: Class label (only for multiclass models). For objects of class `xgboost` (as
#'   produced by [xgboost()]), it will be a `factor`, while for objects of class `xgb.Booster`
#'   (as produced by [xgb.train()]), it will be a zero-based integer vector.
#'
#' If `feature_names` is not provided and `model` doesn't have `feature_names`,
#' the index of the features will be used instead. Because the index is extracted from the model dump
#' (based on C++ code), it starts at 0 (as in C/C++ or Python) instead of 1 (usual in R).
#'
#' @examples
#' # binary classification using "gbtree":
#' data("ToothGrowth")
#' x <- ToothGrowth[, c("len", "dose")]
#' y <- ToothGrowth$supp
#' model_tree_binary <- xgboost(
#'   x, y,
#'   nrounds = 5L,
#'   nthreads = 1L,
#'   booster = "gbtree",
#'   max_depth = 2L
#' )
#' xgb.importance(model_tree_binary)
#'
#' # binary classification using "gblinear":
#' model_tree_linear <- xgboost(
#'   x, y,
#'   nrounds = 5L,
#'   nthreads = 1L,
#'   booster = "gblinear",
#'   learning_rate = 0.3
#' )
#' xgb.importance(model_tree_linear)
#'
#' # multi-class classification using "gbtree":
#' data("iris")
#' x <- iris[, c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")]
#' y <- iris$Species
#' model_tree_multi <- xgboost(
#'   x, y,
#'   nrounds = 5L,
#'   nthreads = 1L,
#'   booster = "gbtree",
#'   max_depth = 3
#' )
#' # all classes clumped together:
#' xgb.importance(model_tree_multi)
#' # inspect importances separately for each class:
#' num_classes <- 3L
#' nrounds <- 5L
#' xgb.importance(
#'   model_tree_multi, trees = seq(from = 1, by = num_classes, length.out = nrounds)
#' )
#' xgb.importance(
#'   model_tree_multi, trees = seq(from = 2, by = num_classes, length.out = nrounds)
#' )
#' xgb.importance(
#'   model_tree_multi, trees = seq(from = 3, by = num_classes, length.out = nrounds)
#' )
#'
#' # multi-class classification using "gblinear":
#' model_linear_multi <- xgboost(
#'   x, y,
#'   nrounds = 5L,
#'   nthreads = 1L,
#'   booster = "gblinear",
#'   learning_rate = 0.2
#' )
#' xgb.importance(model_linear_multi)
#' @export
xgb.importance <- function(model = NULL, feature_names = getinfo(model, "feature_name"), trees = NULL) {

  if (!(is.null(feature_names) || is.character(feature_names)))
    stop("feature_names: Has to be a character vector")

  if (!is.null(trees)) {
    if (!is.vector(trees)) {
      stop("'trees' must be a vector of tree indices.")
    }
    trees <- trees - 1L
    if (anyNA(trees)) {
      stop("Passed invalid tree indices.")
    }
  }

  handle <- xgb.get.handle(model)
  if (xgb.booster_type(model) == "gblinear") {
    args <- list(importance_type = "weight", feature_names = feature_names)
    results <- .Call(
      XGBoosterFeatureScore_R, handle, jsonlite::toJSON(args, auto_unbox = TRUE, null = "null")
    )
    names(results) <- c("features", "shape", "weight")
    if (length(results$shape) == 2) {
        n_classes <- results$shape[2]
    } else {
        n_classes <- 0
    }
    importance <- if (n_classes == 0) {
      return(data.table(Feature = results$features, Weight = results$weight)[order(-abs(Weight))])
    } else {
      out <- data.table(
        Feature = rep(results$features, each = n_classes), Weight = results$weight, Class = seq_len(n_classes) - 1
      )[order(Class, -abs(Weight))]
      if (inherits(model, "xgboost") && NROW(attributes(model)$metadata$y_levels)) {
        class_vec <- out$Class
        class_vec <- as.integer(class_vec) + 1L
        attributes(class_vec)$levels <- attributes(model)$metadata$y_levels
        attributes(class_vec)$class <- "factor"
        out[, Class := class_vec]
      }
      return(out[])
    }
  } else {
    concatenated <- list()
    output_names <- vector()
    for (importance_type in c("weight", "total_gain", "total_cover")) {
      args <- list(importance_type = importance_type, feature_names = feature_names, tree_idx = trees)
      results <- .Call(
        XGBoosterFeatureScore_R, handle, jsonlite::toJSON(args, auto_unbox = TRUE, null = "null")
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
