#' eXtreme Gradient Boosting Training
#' 
#' \code{xgb.train} is an advanced interface for training an xgboost model.
#' The \code{xgboost} function is a simpler wrapper for \code{xgb.train}.
#'
#' @param params the list of parameters. 
#'        The complete list of parameters is available at \url{http://xgboost.readthedocs.io/en/latest/parameter.html}.
#'        Below is a shorter summary:
#' 
#' 1. General Parameters
#' 
#' \itemize{
#'   \item \code{booster} which booster to use, can be \code{gbtree} or \code{gblinear}. Default: \code{gbtree}.
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
#'   \item \code{min_child_weight} minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node. The larger, the more conservative the algorithm will be. Default: 1
#'   \item \code{subsample} subsample ratio of the training instance. Setting it to 0.5 means that xgboost randomly collected half of the data instances to grow trees and this will prevent overfitting. It makes computation shorter (because less data to analyse). It is advised to use this parameter with \code{eta} and increase \code{nround}. Default: 1 
#'   \item \code{colsample_bytree} subsample ratio of columns when constructing each tree. Default: 1
#'   \item \code{num_parallel_tree} Experimental parameter. number of trees to grow per round. Useful to test Random Forest through Xgboost (set \code{colsample_bytree < 1}, \code{subsample  < 1}  and \code{round = 1}) accordingly. Default: 1
#'   \item \code{monotone_constraints} A numerical vector consists of \code{1}, \code{0} and \code{-1} with its length equals to the number of features in the training data. \code{1} is increasing, \code{-1} is decreasing and \code{0} is no constraint.
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
#'     \item \code{multi:softmax} set xgboost to do multiclass classification using the softmax objective. Class is represented by a number and should be from 0 to \code{num_class - 1}.
#'     \item \code{multi:softprob} same as softmax, but prediction outputs a vector of ndata * nclass elements, which can be further reshaped to ndata, nclass matrix. The result contains predicted probabilities of each data point belonging to each class.
#'     \item \code{rank:pairwise} set xgboost to do ranking task by minimizing the pairwise loss.
#'   }
#'   \item \code{base_score} the initial prediction score of all instances, global bias. Default: 0.5
#'   \item \code{eval_metric} evaluation metrics for validation data. Users can pass a self-defined function to it. Default: metric will be assigned according to objective(rmse for regression, and error for classification, mean average precision for ranking). List is provided in detail section.
#' }
#' 
#' @param data training dataset. \code{xgb.train} accepts only an \code{xgb.DMatrix} as the input.
#'        \code{xgboost}, in addition, also accepts \code{matrix}, \code{dgCMatrix}, or name of a local data file.
#' @param nrounds max number of boosting iterations.
#' @param watchlist named list of xgb.DMatrix datasets to use for evaluating model performance.
#'        Metrics specified in either \code{eval_metric} or \code{feval} will be computed for each
#'        of these datasets during each boosting iteration, and stored in the end as a field named 
#'        \code{evaluation_log} in the resulting object. When either \code{verbose>=1} or 
#'        \code{\link{cb.print.evaluation}} callback is engaged, the performance results are continuously
#'        printed out during the training. 
#'        E.g., specifying \code{watchlist=list(validation1=mat1, validation2=mat2)} allows to track
#'        the performance of each round's model on mat1 and mat2.
#' @param obj customized objective function. Returns gradient and second order 
#'        gradient with given prediction and dtrain.
#' @param feval custimized evaluation function. Returns 
#'        \code{list(metric='metric-name', value='metric-value')} with given 
#'        prediction and dtrain.
#' @param verbose If 0, xgboost will stay silent. If 1, it will print information about performance.
#'        If 2, some additional information will be printed out.
#'        Note that setting \code{verbose > 0} automatically engages the 
#'        \code{cb.print.evaluation(period=1)} callback function.
#' @param print_every_n Print each n-th iteration evaluation messages when \code{verbose>0}.
#'        Default is 1 which means all messages are printed. This parameter is passed to the 
#'        \code{\link{cb.print.evaluation}} callback.
#' @param early_stopping_rounds If \code{NULL}, the early stopping function is not triggered. 
#'        If set to an integer \code{k}, training with a validation set will stop if the performance 
#'        doesn't improve for \code{k} rounds.
#'        Setting this parameter engages the \code{\link{cb.early.stop}} callback.
#' @param maximize If \code{feval} and \code{early_stopping_rounds} are set,
#'        then this parameter must be set as well.
#'        When it is \code{TRUE}, it means the larger the evaluation score the better.
#'        This parameter is passed to the \code{\link{cb.early.stop}} callback.
#' @param save_period when it is non-NULL, model is saved to disk after every \code{save_period} rounds,
#'        0 means save at the end. The saving is handled by the \code{\link{cb.save.model}} callback.
#' @param save_name the name or path for periodically saved model file.
#' @param xgb_model a previously built model to continue the training from.
#'        Could be either an object of class \code{xgb.Booster}, or its raw data, or the name of a 
#'        file with a previously saved model.
#' @param callbacks a list of callback functions to perform various task during boosting.
#'        See \code{\link{callbacks}}. Some of the callbacks are automatically created depending on the 
#'        parameters' values. User can provide either existing or their own callback methods in order 
#'        to customize the training process.
#' @param ... other parameters to pass to \code{params}.
#' @param label vector of response values. Should not be provided when data is 
#'        a local data file name or an \code{xgb.DMatrix}.
#' @param missing by default is set to NA, which means that NA values should be considered as 'missing'
#'        by the algorithm. Sometimes, 0 or other extreme value might be used to represent missing values.
#'        This parameter is only used when input is a dense matrix.
#' @param weight a vector indicating the weight for each row of the input.
#' 
#' @details 
#' These are the training functions for \code{xgboost}. 
#' 
#' The \code{xgb.train} interface supports advanced features such as \code{watchlist}, 
#' customized objective and evaluation metric functions, therefore it is more flexible 
#' than the \code{xgboost} interface.
#'
#' Parallelization is automatically enabled if \code{OpenMP} is present. 
#' Number of threads can also be manually specified via \code{nthread} parameter.
#' 
#' The evaluation metric is chosen automatically by Xgboost (according to the objective)
#' when the \code{eval_metric} parameter is not provided.
#' User may set one or several \code{eval_metric} parameters. 
#' Note that when using a customized metric, only this single metric can be used.
#' The folloiwing is the list of built-in metrics for which Xgboost provides optimized implementation:
#'   \itemize{
#'      \item \code{rmse} root mean square error. \url{http://en.wikipedia.org/wiki/Root_mean_square_error}
#'      \item \code{logloss} negative log-likelihood. \url{http://en.wikipedia.org/wiki/Log-likelihood}
#'      \item \code{mlogloss} multiclass logloss. \url{https://www.kaggle.com/wiki/MultiClassLogLoss/}
#'      \item \code{error} Binary classification error rate. It is calculated as \code{(# wrong cases) / (# all cases)}.
#'            By default, it uses the 0.5 threshold for predicted values to define negative and positive instances.
#'            Different threshold (e.g., 0.) could be specified as "error@0."
#'      \item \code{merror} Multiclass classification error rate. It is calculated as \code{(# wrong cases) / (# all cases)}.
#'      \item \code{auc} Area under the curve. \url{http://en.wikipedia.org/wiki/Receiver_operating_characteristic#'Area_under_curve} for ranking evaluation.
#'      \item \code{ndcg} Normalized Discounted Cumulative Gain (for ranking task). \url{http://en.wikipedia.org/wiki/NDCG}
#'   }
#' 
#' The following callbacks are automatically created when certain parameters are set:
#' \itemize{
#'   \item \code{cb.print.evaluation} is turned on when \code{verbose > 0};
#'         and the \code{print_every_n} parameter is passed to it.
#'   \item \code{cb.evaluation.log} is on when \code{watchlist} is present.
#'   \item \code{cb.early.stop}: when \code{early_stopping_rounds} is set.
#'   \item \code{cb.save.model}: when \code{save_period > 0} is set.
#' }
#' 
#' @return 
#' An object of class \code{xgb.Booster} with the following elements:
#' \itemize{
#'   \item \code{handle} a handle (pointer) to the xgboost model in memory.
#'   \item \code{raw} a cached memory dump of the xgboost model saved as R's \code{raw} type.
#'   \item \code{niter} number of boosting iterations.
#'   \item \code{evaluation_log} evaluation history storead as a \code{data.table} with the
#'         first column corresponding to iteration number and the rest corresponding to evaluation
#'         metrics' values. It is created by the \code{\link{cb.evaluation.log}} callback.
#'   \item \code{call} a function call.
#'   \item \code{params} parameters that were passed to the xgboost library. Note that it does not 
#'         capture parameters changed by the \code{\link{cb.reset.parameters}} callback.
#'   \item \code{callbacks} callback functions that were either automatically assigned or 
#'         explicitely passed.
#'   \item \code{best_iteration} iteration number with the best evaluation metric value
#'         (only available with early stopping).
#'   \item \code{best_ntreelimit} the \code{ntreelimit} value corresponding to the best iteration, 
#'         which could further be used in \code{predict} method
#'         (only available with early stopping).
#'   \item \code{best_score} the best evaluation metric value during early stopping.
#'         (only available with early stopping).
#'   \item \code{feature_names} names of the training dataset features
#'         (only when comun names were defined in training data).
#' }
#' 
#' @seealso
#' \code{\link{callbacks}},
#' \code{\link{predict.xgb.Booster}},
#' \code{\link{xgb.cv}}
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' 
#' dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
#' dtest <- xgb.DMatrix(agaricus.test$data, label = agaricus.test$label)
#' watchlist <- list(train = dtrain, eval = dtest)
#' 
#' ## A simple xgb.train example:
#' param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2, 
#'               objective = "binary:logistic", eval_metric = "auc")
#' bst <- xgb.train(param, dtrain, nrounds = 2, watchlist)
#' 
#' 
#' ## An xgb.train example where custom objective and evaluation metric are used:
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
#' 
#' # These functions could be used by passing them either:
#' #  as 'objective' and 'eval_metric' parameters in the params list:
#' param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2, 
#'               objective = logregobj, eval_metric = evalerror)
#' bst <- xgb.train(param, dtrain, nrounds = 2, watchlist)
#' 
#' #  or through the ... arguments:
#' param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2)
#' bst <- xgb.train(param, dtrain, nrounds = 2, watchlist,
#'                  objective = logregobj, eval_metric = evalerror)
#' 
#' #  or as dedicated 'obj' and 'feval' parameters of xgb.train:
#' bst <- xgb.train(param, dtrain, nrounds = 2, watchlist,
#'                  obj = logregobj, feval = evalerror)
#' 
#' 
#' ## An xgb.train example of using variable learning rates at each iteration:
#' param <- list(max_depth = 2, eta = 1, silent = 1, nthread = 2,
#'               objective = "binary:logistic", eval_metric = "auc")
#' my_etas <- list(eta = c(0.5, 0.1))
#' bst <- xgb.train(param, dtrain, nrounds = 2, watchlist,
#'                  callbacks = list(cb.reset.parameters(my_etas)))
#' 
#' ## Early stopping:
#' bst <- xgb.train(param, dtrain, nrounds = 25, watchlist,
#'                  early_stopping_rounds = 3)
#' 
#' ## An 'xgboost' interface example:
#' bst <- xgboost(data = agaricus.train$data, label = agaricus.train$label, 
#'                max_depth = 2, eta = 1, nthread = 2, nrounds = 2, 
#'                objective = "binary:logistic")
#' pred <- predict(bst, agaricus.test$data)
#' 
#' @rdname xgb.train
#' @export
xgb.train <- function(params = list(), data, nrounds, watchlist = list(),
                      obj = NULL, feval = NULL, verbose = 1, print_every_n = 1L,
                      early_stopping_rounds = NULL, maximize = NULL,
                      save_period = NULL, save_name = "xgboost.model", 
                      xgb_model = NULL, callbacks = list(), ...) {
  
  check.deprecation(...)
  
  params <- check.booster.params(params, ...)

  check.custom.obj()
  check.custom.eval()
  
  # data & watchlist checks
  dtrain <- data
  if (!inherits(dtrain, "xgb.DMatrix")) 
    stop("second argument dtrain must be xgb.DMatrix")
  if (length(watchlist) > 0) {
    if (typeof(watchlist) != "list" ||
        !all(vapply(watchlist, inherits, logical(1), what = 'xgb.DMatrix')))
      stop("watchlist must be a list of xgb.DMatrix elements")
    evnames <- names(watchlist)
    if (is.null(evnames) || any(evnames == ""))
      stop("each element of the watchlist must have a name tag")
  }

  # evaluation printing callback
  params <- c(params, list(silent = ifelse(verbose > 1, 0, 1)))
  print_every_n <- max( as.integer(print_every_n), 1L)
  if (!has.callbacks(callbacks, 'cb.print.evaluation') &&
      verbose) {
    callbacks <- add.cb(callbacks, cb.print.evaluation(print_every_n))
  }
  # evaluation log callback:  it is automatically enabled when watchlist is provided
  evaluation_log <- list()
  if (!has.callbacks(callbacks, 'cb.evaluation.log') &&
      length(watchlist) > 0) {
    callbacks <- add.cb(callbacks, cb.evaluation.log())
  }
  # Model saving callback
  if (!is.null(save_period) &&
      !has.callbacks(callbacks, 'cb.save.model')) {
    callbacks <- add.cb(callbacks, cb.save.model(save_period, save_name))
  }
  # Early stopping callback
  stop_condition <- FALSE
  if (!is.null(early_stopping_rounds) &&
      !has.callbacks(callbacks, 'cb.early.stop')) {
    callbacks <- add.cb(callbacks, cb.early.stop(early_stopping_rounds, 
                                                 maximize = maximize, verbose = verbose))
  }
  # Sort the callbacks into categories
  cb <- categorize.callbacks(callbacks)

  # The tree updating process would need slightly different handling
  is_update <- NVL(params[['process_type']], '.') == 'update'

  # Construct a booster (either a new one or load from xgb_model)
  handle <- xgb.Booster.handle(params, append(watchlist, dtrain), xgb_model)
  bst <- xgb.handleToBooster(handle)

  # extract parameters that can affect the relationship b/w #trees and #iterations
  num_class <- max(as.numeric(NVL(params[['num_class']], 1)), 1)
  num_parallel_tree <- max(as.numeric(NVL(params[['num_parallel_tree']], 1)), 1)

  # When the 'xgb_model' was set, find out how many boosting iterations it has
  niter_init <- 0
  if (!is.null(xgb_model)) {
    niter_init <- as.numeric(xgb.attr(bst, 'niter')) + 1
    if (length(niter_init) == 0) {
      niter_init <- xgb.ntree(bst) %/% (num_parallel_tree * num_class)
    }
  }
  if(is_update && nrounds > niter_init)
    stop("nrounds cannot be larger than ", niter_init, " (nrounds of xgb_model)")

  # TODO: distributed code
  rank <- 0
  
  niter_skip <- ifelse(is_update, 0, niter_init)
  begin_iteration <- niter_skip + 1
  end_iteration <- niter_skip + nrounds
  
  # the main loop for boosting iterations
  for (iteration in begin_iteration:end_iteration) {
    
    for (f in cb$pre_iter) f()
    
    xgb.iter.update(bst$handle, dtrain, iteration - 1, obj)
    
    bst_evaluation <- numeric(0)
    if (length(watchlist) > 0)
      bst_evaluation <- xgb.iter.eval(bst$handle, watchlist, iteration - 1, feval)
    
    xgb.attr(bst$handle, 'niter') <- iteration - 1

    for (f in cb$post_iter) f()

    if (stop_condition) break
  }
  for (f in cb$finalize) f(finalize = TRUE)
  
  bst <- xgb.Booster.complete(bst, saveraw = TRUE)
  
  # store the total number of boosting iterations
  bst$niter = end_iteration

  # store the evaluation results
  if (length(evaluation_log) > 0 &&
      nrow(evaluation_log) > 0) {
    # include the previous compatible history when available
    if (inherits(xgb_model, 'xgb.Booster') &&
        !is_update &&
        !is.null(xgb_model$evaluation_log) &&
        all.equal(colnames(evaluation_log),
                  colnames(xgb_model$evaluation_log))) {
      evaluation_log <- rbindlist(list(xgb_model$evaluation_log, evaluation_log))
    }
    bst$evaluation_log <- evaluation_log
  }

  bst$call <- match.call()
  bst$params <- params
  bst$callbacks <- callbacks
  if (!is.null(colnames(dtrain)))
    bst$feature_names <- colnames(dtrain)
  
  return(bst)
}
