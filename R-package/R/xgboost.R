#' eXtreme Gradient Boosting (Tree) library
#' 
#' A simple interface for xgboost in R
#' 
#' @param data takes \code{matrix}, \code{dgCMatrix}, local data file or 
#'   \code{xgb.DMatrix}. 
#' @param label the response variable. User should not set this field,
#'    if data is local data file or  \code{xgb.DMatrix}. 
#' @param params the list of parameters.
#' 
#' 1. General Parameters
#' 
#' \itemize{
#'   \item \code{booster} which booster to use, can be \code{gbtree} or \code{gblinear}. Default: \code{gbtree}
#'   \item \code{silent} 0 means printing running messages, 1 means silent mode. Default: 0
#'   \item \code{nthread} number of parallel threads used to run xgboost. Default to maximum number of threads available if not set.
#'   \item \code{num_pbuffer} size of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step. Default: set automatically by xgboost, no need to be set by user
#'   \item \code{num_feature} feature dimension used in boosting, set to maximum dimension of the feature. Default: set automatically by xgboost, no need to be set by user.
#' }
#'  
#' 2. Booster Parameters
#' 
#' 2.1. Parameter for Tree Booster
#' 
#' \itemize{
#'   \item \code{eta} step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features. and eta actually shrinkage the feature weights to make the boosting process more conservative. Default: 0.3
#'   \item \code{gamma} minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, the more conservative the algorithm will be. 
#'   \item \code{max_depth} maximum depth of a tree. Default: 6
#'   \item \code{min_child_weight} minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. Default: 1
#'   \item \code{subsample} subsample ratio of the training instance. Setting it to 0.5 means that xgboost randomly collected half of the data instances to grow trees and this will prevent overfitting. Default: 1
#'   \item \code{colsample_bytree} subsample ratio of columns when constructing each tree. Default: 1
#' }
#' 
#' 2.2. Parameter for Linear Booster
#'  
#' \itemize{
#'   \item \code{lambda} L2 regularization term on weights. Default: 0
#'   \item \code{lambda_bias} L2 regularization term on bias. Default: 0
#'   \item \code{alpha} L1 regularization term on weights. (there is no L1 reg on bias because it is not important). Default: 0
#' }
#' 
#' 3. Task Parameters 
#' 
#' \itemize{
#' \item \code{objective} specify the learning task and the corresponding learning objective, and the objective options are below:
#'   \itemize{
#'     \item \code{reg:linear} linear regression (Default).
#'     \item \code{reg:logistic} logistic regression.
#'     \item \code{binary:logistic} logistic regression for binary classification. Output probability.
#'     \item \code{binary:logitraw} logistic regression for binary classification, output score before logistic transformation.
#'     \item \code{multi:softmax} set xgboost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes).
#'     \item \code{multi:softprob} same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
#'     \item \code{rank:pairwise} set xgboost to do ranking task by minimizing the pairwise loss.
#'   }
#'   \item \code{base_score} the initial prediction score of all instances, global bias. Default: 0.5
#'   \item \code{eval_metric} evaluation metrics for validation data, a default metric will be assigned according to objective(rmse for regression, and error for classification, mean average precision for ranking). Default according to objective. The choices are listed below:
#'   \itemize{
#'      \item \code{rmse} root mean square error. \url{http://en.wikipedia.org/wiki/Root_mean_square_error}
#'      \item \code{logloss} negative log-likelihood. \url{http://en.wikipedia.org/wiki/Log-likelihood}
#'      \item \code{error} Binary classification error rate. It is calculated as \code{(wrong cases) / (all cases)}. For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
#'      \item \code{merror} Multiclass classification error rate. It is calculated as \code{(wrong cases) / (all cases)}.
#'      \item \code{auc} Area under the curve. \url{http://en.wikipedia.org/wiki/Receiver_operating_characteristic#'Area_under_curve} for ranking evaluation.
#'      \item \code{ndcg} Normalized Discounted Cumulative Gain. \url{http://en.wikipedia.org/wiki/NDCG}
#'   }
#'   \item \code{map} Mean average precision. \url{http://en.wikipedia.org/wiki/Mean_average_precision#'Mean_average_precision}
#'   \item \code{ndcg@@n} and \code{map@@n} n can be assigned as an integer to cut off the top positions in the lists for evaluation.
#'   \item \code{ndcg-}, \code{map-}, \code{ndcg@@n-}, \code{map@@n-} In xgboost, NDCG and MAP will evaluate the score of a list without any positive samples as 1. By adding "-" in the evaluation metric xgboost will evaluate these score as 0 to be consistent under some conditions. Training repeatively.
#'   \item \code{seed} random number seed. Default: 0
#' }
#' 
#' @param nrounds the max number of iterations
#' @param verbose If 0, xgboost will stay silent. If 1, xgboost will print 
#'   information of performance. If 2, xgboost will print information of both
#'   performance and construction progress information
#' @param missing Missing is only used when input is dense matrix, pick a float 
#'     value that represents missing value. Sometime a data use 0 or other extreme value to represents missing values.
#' @param ... other parameters to pass to \code{params}.
#' 
#' @details 
#' This is the modeling function for xgboost.
#' 
#' Parallelization is automatically enabled if OpenMP is present.
#' Number of threads can also be manually specified via "nthread" parameter
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' train <- agaricus.train
#' test <- agaricus.test
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nround = 2,objective = "binary:logistic")
#' pred <- predict(bst, test$data)
#' 
#' @export
#' 
xgboost <- function(data = NULL, label = NULL, missing = NULL, params = list(), nrounds, 
                    verbose = 1, ...) {
  if (is.null(missing)) {
    dtrain <- xgb.get.DMatrix(data, label)
  } else {
    dtrain <- xgb.get.DMatrix(data, label, missing)
  }
    
  params <- append(params, list(...))
  
  if (verbose > 0) {
    watchlist <- list(train = dtrain)
  } else {
    watchlist <- list()
  }
  
  bst <- xgb.train(params, dtrain, nrounds, watchlist, verbose=verbose)
  
  return(bst)
} 


#' Training part from Mushroom Data Set
#' 
#' This data set is originally from the Mushroom data set,
#' UCI Machine Learning Repository.
#' 
#' This data set includes the following fields:
#' 
#' \itemize{
#'  \item \code{label} the label for each record
#'  \item \code{data} a sparse Matrix of \code{dgCMatrix} class, with 126 columns.
#' }
#'
#' @references
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#' 
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository 
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
#' School of Information and Computer Science.
#' 
#' @docType data
#' @keywords datasets
#' @name agaricus.train
#' @usage data(agaricus.train)
#' @format A list containing a label vector, and a dgCMatrix object with 6513 
#' rows and 127 variables
NULL

#' Test part from Mushroom Data Set
#'
#' This data set is originally from the Mushroom data set,
#' UCI Machine Learning Repository.
#' 
#' This data set includes the following fields:
#' 
#' \itemize{
#'  \item \code{label} the label for each record
#'  \item \code{data} a sparse Matrix of \code{dgCMatrix} class, with 126 columns.
#' }
#'
#' @references
#' https://archive.ics.uci.edu/ml/datasets/Mushroom
#' 
#' Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository 
#' [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
#' School of Information and Computer Science.
#' 
#' @docType data
#' @keywords datasets
#' @name agaricus.test
#' @usage data(agaricus.test)
#' @format A list containing a label vector, and a dgCMatrix object with 1611 
#' rows and 126 variables
NULL
