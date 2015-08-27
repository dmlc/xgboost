#' eXtreme Gradient Boosting Training
#' 
#' An advanced interface for training xgboost model. Look at \code{\link{xgboost}} function for a simpler interface.
#'
#' @param params the list of parameters. 
#' 
#' 1. General Parameters
#' 
#' \itemize{
#'   \item \code{booster} which booster to use, can be \code{gbtree} or \code{gblinear}. Default: \code{gbtree}
#'   \item \code{silent} 0 means printing running messages, 1 means silent mode. Default: 0
#' }
#'  
#' 2. Booster Parameters
#' 
#' 2.1. Parameter for Tree Booster
#' 
#' \itemize{
#'   \item \code{eta} control the learning rate: scale the contribution of each tree by a factor of \code{0 < eta < 1} when it is added to the current approximation. Used to prevent overfitting by making the boosting process more conservative. Lower value for \code{eta} implies larger value for \code{nrounds}: low \code{eta} value means model more robust to overfitting but slower to compute. Default: 0.3
#'   \item \code{gamma} minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, the more conservative the algorithm will be. 
#'   \item \code{max_depth} maximum depth of a tree. Default: 6
#'   \item \code{min_child_weight} minimum sum of instance weight(hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. Default: 1
#'   \item \code{subsample} subsample ratio of the training instance. Setting it to 0.5 means that xgboost randomly collected half of the data instances to grow trees and this will prevent overfitting. It makes computation shorter (because less data to analyse). It is advised to use this parameter with \code{eta} and increase \code{nround}. Default: 1 
#'   \item \code{colsample_bytree} subsample ratio of columns when constructing each tree. Default: 1
#'   \item \code{num_parallel_tree} Experimental parameter. number of trees to grow per round. Useful to test Random Forest through Xgboost (set \code{colsample_bytree < 1}, \code{subsample  < 1}  and \code{round = 1}) accordingly. Default: 1
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
#' \item \code{objective} specify the learning task and the corresponding learning objective, users can pass a self-defined function to it. The default objective options are below:
#'   \itemize{
#'     \item \code{reg:linear} linear regression (Default).
#'     \item \code{reg:logistic} logistic regression.
#'     \item \code{binary:logistic} logistic regression for binary classification. Output probability.
#'     \item \code{binary:logitraw} logistic regression for binary classification, output score before logistic transformation.
#'     \item \code{num_class} set the number of classes. To use only with multiclass objectives.
#'     \item \code{multi:softmax} set xgboost to do multiclass classification using the softmax objective. Class is represented by a number and should be from 0 to \code{tonum_class}.
#'     \item \code{multi:softprob} same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probabilities of each data point belonging to each class.
#'     \item \code{rank:pairwise} set xgboost to do ranking task by minimizing the pairwise loss.
#'   }
#'   \item \code{base_score} the initial prediction score of all instances, global bias. Default: 0.5
#'   \item \code{eval_metric} evaluation metrics for validation data. Users can pass a self-defined function to it. Default: metric will be assigned according to objective(rmse for regression, and error for classification, mean average precision for ranking). List is provided in detail section.
#' }
#' 
#' @param data takes an \code{xgb.DMatrix} as the input.
#' @param nrounds the max number of iterations
#' @param watchlist what information should be printed when \code{verbose=1} or
#'   \code{verbose=2}. Watchlist is used to specify validation set monitoring
#'   during training. For example user can specify
#'    watchlist=list(validation1=mat1, validation2=mat2) to watch
#'    the performance of each round's model on mat1 and mat2
#'
#' @param obj customized objective function. Returns gradient and second order 
#'   gradient with given prediction and dtrain, 
#' @param feval custimized evaluation function. Returns 
#'   \code{list(metric='metric-name', value='metric-value')} with given 
#'   prediction and dtrain,
#' @param verbose If 0, xgboost will stay silent. If 1, xgboost will print 
#'   information of performance. If 2, xgboost will print information of both
#' @param print.every.n Print every N progress messages when \code{verbose>0}. Default is 1 which means all messages are printed.
#' @param early.stop.round If \code{NULL}, the early stopping function is not triggered. 
#'     If set to an integer \code{k}, training with a validation set will stop if the performance 
#'     keeps getting worse consecutively for \code{k} rounds.
#' @param maximize If \code{feval} and \code{early.stop.round} are set, then \code{maximize} must be set as well.
#'     \code{maximize=TRUE} means the larger the evaluation score the better.
#' @param save_period save the model to the disk in every \code{save_period} rounds, 0 means no such action.
#' @param save_name the name or path for periodically saved model file.
#' @param ... other parameters to pass to \code{params}.
#' 
#' @details 
#' This is the training function for \code{xgboost}. 
#' 
#' It supports advanced features such as \code{watchlist}, customized objective function (\code{feval}),
#' therefore it is more flexible than \code{\link{xgboost}} function.
#'
#' Parallelization is automatically enabled if \code{OpenMP} is present. 
#' Number of threads can also be manually specified via \code{nthread} parameter.
#' 
#' \code{eval_metric} parameter (not listed above) is set automatically by Xgboost but can be overriden by parameter. Below is provided the list of different metric optimized by Xgboost to help you to understand how it works inside or to use them with the \code{watchlist} parameter.
#'   \itemize{
#'      \item \code{rmse} root mean square error. \url{http://en.wikipedia.org/wiki/Root_mean_square_error}
#'      \item \code{logloss} negative log-likelihood. \url{http://en.wikipedia.org/wiki/Log-likelihood}
#'      \item \code{error} Binary classification error rate. It is calculated as \code{(wrong cases) / (all cases)}. For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
#'      \item \code{merror} Multiclass classification error rate. It is calculated as \code{(wrong cases) / (all cases)}.
#'      \item \code{auc} Area under the curve. \url{http://en.wikipedia.org/wiki/Receiver_operating_characteristic#'Area_under_curve} for ranking evaluation.
#'      \item \code{ndcg} Normalized Discounted Cumulative Gain (for ranking task). \url{http://en.wikipedia.org/wiki/NDCG}
#'   }
#'   
#' Full list of parameters is available in the Wiki \url{https://github.com/dmlc/xgboost/wiki/Parameters}.
#' 
#' This function only accepts an \code{\link{xgb.DMatrix}} object as the input.
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
#' dtest <- dtrain
#' watchlist <- list(eval = dtest, train = dtrain)
#' logregobj <- function(preds, dtrain) {
#'    labels <- getinfo(dtrain, "label")
#'    preds <- 1/(1 + exp(-preds))
#'    grad <- preds - labels
#'    hess <- preds * (1 - preds)
#'    return(list(grad = grad, hess = hess))
#' }
#' evalerror <- function(preds, dtrain) {
#'   labels <- getinfo(dtrain, "label")
#'   err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
#'   return(list(metric = "error", value = err))
#' }
#' param <- list(max.depth = 2, eta = 1, silent = 1, objective=logregobj,eval_metric=evalerror)
#' bst <- xgb.train(param, dtrain, nthread = 2, nround = 2, watchlist)
#' @export
#' 
xgb.train <- function(params=list(), data, nrounds, watchlist = list(), 
                      obj = NULL, feval = NULL, verbose = 1, print.every.n=1L,
                      early.stop.round = NULL, maximize = NULL, 
                      save_period = 0, save_name = "xgboost.model", ...) {
  dtrain <- data
  if (typeof(params) != "list") {
    stop("xgb.train: first argument params must be list")
  }
  if (class(dtrain) != "xgb.DMatrix") {
    stop("xgb.train: second argument dtrain must be xgb.DMatrix")
  }
  if (verbose > 1) {
    params <- append(params, list(silent = 0))
  } else {
    params <- append(params, list(silent = 1))
  }
  if (length(watchlist) != 0 && verbose == 0) {
    warning('watchlist is provided but verbose=0, no evaluation information will be printed')
  }
  
  dot.params = list(...)
  nms.params = names(params)
  nms.dot.params = names(dot.params)
  if (length(intersect(nms.params,nms.dot.params))>0)
    stop("Duplicated term in parameters. Please check your list of params.")
  params = append(params, dot.params)
  
  # customized objective and evaluation metric interface
  if (!is.null(params$objective) && !is.null(obj))
    stop("xgb.train: cannot assign two different objectives")
  if (!is.null(params$objective))
    if (class(params$objective)=='function') {
      obj = params$objective
      params$objective = NULL
    }
  if (!is.null(params$eval_metric) && !is.null(feval))
    stop("xgb.train: cannot assign two different evaluation metrics")
  if (!is.null(params$eval_metric))
    if (class(params$eval_metric)=='function') {
      feval = params$eval_metric
      params$eval_metric = NULL
    }
    
  # Early stopping
  if (!is.null(early.stop.round)){
    if (!is.null(feval) && is.null(maximize))
      stop('Please set maximize to note whether the model is maximizing the evaluation or not.')
    if (length(watchlist) == 0)
      stop('For early stopping you need at least one set in watchlist.')
    if (is.null(maximize) && is.null(params$eval_metric))
      stop('Please set maximize to note whether the model is maximizing the evaluation or not.')
    if (is.null(maximize))
    {
      if (params$eval_metric %in% c('rmse','logloss','error','merror','mlogloss')) {
        maximize = FALSE
      } else {
        maximize = TRUE
      }
    }
    
    if (maximize) {
      bestScore = 0
    } else {
      bestScore = Inf
    }
    bestInd = 0
    earlyStopflag = FALSE
    
    if (length(watchlist)>1)
      warning('Only the first data set in watchlist is used for early stopping process.')
  }
  
  
  handle <- xgb.Booster(params, append(watchlist, dtrain))
  bst <- xgb.handleToBooster(handle)
  print.every.n=max( as.integer(print.every.n), 1L)
  for (i in 1:nrounds) {
    succ <- xgb.iter.update(bst$handle, dtrain, i - 1, obj)
    if (length(watchlist) != 0) {
      msg <- xgb.iter.eval(bst$handle, watchlist, i - 1, feval)
      if (0== ( (i-1) %% print.every.n))
	    cat(paste(msg, "\n", sep=""))
      if (!is.null(early.stop.round))
      {
        score = strsplit(msg,':|\\s+')[[1]][3]
        score = as.numeric(score)
        if ((maximize && score>bestScore) || (!maximize && score<bestScore)) {
          bestScore = score
          bestInd = i
        } else {
          if (i-bestInd>=early.stop.round) {
            earlyStopflag = TRUE
            cat('Stopping. Best iteration:',bestInd)
            break
          }
        }
      }
    }
    if (save_period > 0) {
      if (i %% save_period == 0) {
        xgb.save(bst, save_name)
      }
    }
  }
  bst <- xgb.Booster.check(bst)
  if (!is.null(early.stop.round)) {
    bst$bestScore = bestScore
    bst$bestInd = bestInd
  }
  return(bst)
} 
