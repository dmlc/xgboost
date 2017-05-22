#' Create new features from a previously learned model
#' 
#' May improve the learning by adding new features to the training data based on the decision trees from a previously learned model.
#' 
#' @param model decision tree boosting model learned on the original data
#' @param data original data (usually provided as a \code{dgCMatrix} matrix)
#' @param ... currently not used
#' 
#' @return \code{dgCMatrix} matrix including both the original data and the new features.
#'
#' @details 
#' This is the function inspired from the paragraph 3.1 of the paper:
#' 
#' \strong{Practical Lessons from Predicting Clicks on Ads at Facebook}
#' 
#' \emph{(Xinran He, Junfeng Pan, Ou Jin, Tianbing Xu, Bo Liu, Tao Xu, Yan, xin Shi, Antoine Atallah, Ralf Herbrich, Stuart Bowers, 
#' Joaquin Quinonero Candela)}
#'  
#' International Workshop on Data Mining for Online Advertising (ADKDD) - August 24, 2014
#' 
#' \url{https://research.fb.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/}.
#' 
#' Extract explaining the method:
#' 
#' "We found that boosted decision trees are a powerful and very
#' convenient way to implement non-linear and tuple transformations
#' of the kind we just described. We treat each individual
#' tree as a categorical feature that takes as value the
#' index of the leaf an instance ends up falling in. We use 
#' 1-of-K coding of this type of features. 
#' 
#' For example, consider the boosted tree model in Figure 1 with 2 subtrees, 
#' where the first subtree has 3 leafs and the second 2 leafs. If an
#' instance ends up in leaf 2 in the first subtree and leaf 1 in
#' second subtree, the overall input to the linear classifier will
#' be the binary vector \code{[0, 1, 0, 1, 0]}, where the first 3 entries
#' correspond to the leaves of the first subtree and last 2 to
#' those of the second subtree.
#' 
#' [...]
#' 
#' We can understand boosted decision tree
#' based transformation as a supervised feature encoding that
#' converts a real-valued vector into a compact binary-valued
#' vector. A traversal from root node to a leaf node represents
#' a rule on certain features."
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' dtrain <- xgb.DMatrix(data = agaricus.train$data, label = agaricus.train$label)
#' dtest <- xgb.DMatrix(data = agaricus.test$data, label = agaricus.test$label)
#'
#' param <- list(max_depth=2, eta=1, silent=1, objective='binary:logistic')
#' nround = 4
#'
#' bst = xgb.train(params = param, data = dtrain, nrounds = nround, nthread = 2)
#' 
#' # Model accuracy without new features
#' accuracy.before <- sum((predict(bst, agaricus.test$data) >= 0.5) == agaricus.test$label) /
#'                    length(agaricus.test$label)
#' 
#' # Convert previous features to one hot encoding
#' new.features.train <- xgb.create.features(model = bst, agaricus.train$data)
#' new.features.test <- xgb.create.features(model = bst, agaricus.test$data)
#' 
#' # learning with new features
#' new.dtrain <- xgb.DMatrix(data = new.features.train, label = agaricus.train$label)
#' new.dtest <- xgb.DMatrix(data = new.features.test, label = agaricus.test$label)
#' watchlist <- list(train = new.dtrain)
#' bst <- xgb.train(params = param, data = new.dtrain, nrounds = nround, nthread = 2)
#' 
#' # Model accuracy with new features
#' accuracy.after <- sum((predict(bst, new.dtest) >= 0.5) == agaricus.test$label) /
#'                   length(agaricus.test$label)
#' 
#' # Here the accuracy was already good and is now perfect.
#' cat(paste("The accuracy was", accuracy.before, "before adding leaf features and it is now",
#'           accuracy.after, "!\n"))
#' 
#' @export
xgb.create.features <- function(model, data, ...){
  check.deprecation(...)
  pred_with_leaf <- predict(model, data, predleaf = TRUE)
  cols <- lapply(as.data.frame(pred_with_leaf), factor)
  cBind(data, sparse.model.matrix( ~ . -1, cols))
}
