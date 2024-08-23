#' eXtreme Gradient Boosting Training
#'
#' `xgb.train()` is an advanced interface for training an xgboost model.
#' The [xgboost()] function is a simpler wrapper for `xgb.train()`.
#'
#' @param params the list of parameters. The complete list of parameters is
#'   available in the [online documentation](http://xgboost.readthedocs.io/en/latest/parameter.html).
#'   Below is a shorter summary:
#'
#'   **1. General Parameters**
#'
#'   - `booster`: Which booster to use, can be `gbtree` or `gblinear`. Default: `gbtree`.
#'
#'   **2. Booster Parameters**
#'
#'   **2.1. Parameters for Tree Booster**
#'   - `eta`: The learning rate: scale the contribution of each tree by a factor of `0 < eta < 1`
#'     when it is added to the current approximation.
#'     Used to prevent overfitting by making the boosting process more conservative.
#'     Lower value for `eta` implies larger value for `nrounds`: low `eta` value means model
#'     more robust to overfitting but slower to compute. Default: 0.3.
#'   - `gamma`: Minimum loss reduction required to make a further partition on a leaf node of the tree.
#'     the larger, the more conservative the algorithm will be.
#'   - `max_depth`: Maximum depth of a tree. Default: 6.
#'   - `min_child_weight`: Minimum sum of instance weight (hessian) needed in a child.
#'     If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
#'     then the building process will give up further partitioning.
#'     In linear regression mode, this simply corresponds to minimum number of instances needed to be in each node.
#'     The larger, the more conservative the algorithm will be. Default: 1.
#'   - `subsample`: Subsample ratio of the training instance.
#'     Setting it to 0.5 means that xgboost randomly collected half of the data instances to grow trees
#'     and this will prevent overfitting. It makes computation shorter (because less data to analyse).
#'     It is advised to use this parameter with `eta` and increase `nrounds`. Default: 1.
#'   - `colsample_bytree`: Subsample ratio of columns when constructing each tree. Default: 1.
#'   - `lambda`: L2 regularization term on weights. Default: 1.
#'   - `alpha`: L1 regularization term on weights. (there is no L1 reg on bias because it is not important). Default: 0.
#'   - `num_parallel_tree`: Experimental parameter. number of trees to grow per round.
#'     Useful to test Random Forest through XGBoost.
#'     (set `colsample_bytree < 1`, `subsample  < 1` and `round = 1`) accordingly.
#'     Default: 1.
#'   - `monotone_constraints`: A numerical vector consists of `1`, `0` and `-1` with its length
#'     equals to the number of features in the training data.
#'     `1` is increasing, `-1` is decreasing and `0` is no constraint.
#'   - `interaction_constraints`: A list of vectors specifying feature indices of permitted interactions.
#'     Each item of the list represents one permitted interaction where specified features are allowed to interact with each other.
#'     Feature index values should start from `0` (`0` references the first column).
#'     Leave argument unspecified for no interaction constraints.
#'
#'   **2.2. Parameters for Linear Booster**
#'
#'   - `lambda`: L2 regularization term on weights. Default: 0.
#'   - `lambda_bias`: L2 regularization term on bias. Default: 0.
#'   - `alpha`: L1 regularization term on weights. (there is no L1 reg on bias because it is not important). Default: 0.
#'
#'   **3. Task Parameters**
#'
#'   - `objective`: Specifies the learning task and the corresponding learning objective.
#'     users can pass a self-defined function to it. The default objective options are below:
#'     - `reg:squarederror`: Regression with squared loss (default).
#'     - `reg:squaredlogerror`: Regression with squared log loss \eqn{1/2 \cdot (\log(pred + 1) - \log(label + 1))^2}.
#'       All inputs are required to be greater than -1.
#'       Also, see metric rmsle for possible issue with this objective.
#'     - `reg:logistic`: Logistic regression.
#'     - `reg:pseudohubererror`: Regression with Pseudo Huber loss, a twice differentiable alternative to absolute loss.
#'     - `binary:logistic`: Logistic regression for binary classification. Output probability.
#'     - `binary:logitraw`: Logistic regression for binary classification, output score before logistic transformation.
#'     - `binary:hinge`: Hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
#'     - `count:poisson`: Poisson regression for count data, output mean of Poisson distribution.
#'       The parameter `max_delta_step` is set to 0.7 by default in poisson regression
#'       (used to safeguard optimization).
#'     - `survival:cox`: Cox regression for right censored survival time data (negative values are considered right censored).
#'       Note that predictions are returned on the hazard ratio scale (i.e., as HR = exp(marginal_prediction) in the proportional
#'       hazard function \eqn{h(t) = h_0(t) \cdot HR}.
#'     - `survival:aft`: Accelerated failure time model for censored survival time data. See
#'       [Survival Analysis with Accelerated Failure Time](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html)
#'       for details.
#'       The parameter `aft_loss_distribution` specifies the Probability Density Function
#'       used by `survival:aft` and the `aft-nloglik` metric.
#'     - `multi:softmax`: Set xgboost to do multiclass classification using the softmax objective.
#'       Class is represented by a number and should be from 0 to `num_class - 1`.
#'     - `multi:softprob`: Same as softmax, but prediction outputs a vector of ndata * nclass elements, which can be
#'       further reshaped to ndata, nclass matrix. The result contains predicted probabilities of each data point belonging
#'       to each class.
#'     - `rank:pairwise`: Set XGBoost to do ranking task by minimizing the pairwise loss.
#'     - `rank:ndcg`: Use LambdaMART to perform list-wise ranking where
#'       [Normalized Discounted Cumulative Gain (NDCG)](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) is maximized.
#'     - `rank:map`: Use LambdaMART to perform list-wise ranking where
#'       [Mean Average Precision (MAP)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)
#'       is maximized.
#'     - `reg:gamma`: Gamma regression with log-link. Output is a mean of gamma distribution.
#'       It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be
#'       [gamma-distributed](https://en.wikipedia.org/wiki/Gamma_distribution#Applications).
#'     - `reg:tweedie`: Tweedie regression with log-link.
#'       It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be
#'       [Tweedie-distributed](https://en.wikipedia.org/wiki/Tweedie_distribution#Applications).
#'
#'      For custom objectives, one should pass a function taking as input the current predictions (as a numeric
#'      vector or matrix) and the training data (as an `xgb.DMatrix` object) that will return a list with elements
#'      `grad` and `hess`, which should be numeric vectors or matrices with number of rows matching to the numbers
#'      of rows in the training data (same shape as the predictions that are passed as input to the function).
#'      For multi-valued custom objectives, should have shape `[nrows, ntargets]`. Note that negative values of
#'      the Hessian will be clipped, so one might consider using the expected Hessian (Fisher information) if the
#'      objective is non-convex.
#'
#'      See the tutorials [Custom Objective and Evaluation Metric](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html)
#'      and [Advanced Usage of Custom Objectives](https://xgboost.readthedocs.io/en/stable/tutorials/advanced_custom_obj)
#'      for more information about custom objectives.
#'
#'   - `base_score`: The initial prediction score of all instances, global bias. Default: 0.5.
#'   - `eval_metric`: Evaluation metrics for validation data.
#'     Users can pass a self-defined function to it.
#'     Default: metric will be assigned according to objective
#'     (rmse for regression, and error for classification, mean average precision for ranking).
#'     List is provided in detail section.
#' @param data Training dataset. `xgb.train()` accepts only an `xgb.DMatrix` as the input.
#'   [xgboost()], in addition, also accepts `matrix`, `dgCMatrix`, or name of a local data file.
#' @param nrounds Max number of boosting iterations.
#' @param evals Named list of `xgb.DMatrix` datasets to use for evaluating model performance.
#'   Metrics specified in either `eval_metric` or `feval` will be computed for each
#'   of these datasets during each boosting iteration, and stored in the end as a field named
#'   `evaluation_log` in the resulting object. When either `verbose>=1` or
#'   [xgb.cb.print.evaluation()] callback is engaged, the performance results are continuously
#'   printed out during the training.
#'   E.g., specifying `evals=list(validation1=mat1, validation2=mat2)` allows to track
#'   the performance of each round's model on mat1 and mat2.
#' @param obj Customized objective function. Should take two arguments: the first one will be the
#'   current predictions (either a numeric vector or matrix depending on the number of targets / classes),
#'   and the second one will be the `data` DMatrix object that is used for training.
#'
#'   It should return a list with two elements `grad` and `hess` (in that order), as either
#'   numeric vectors or numeric matrices depending on the number of targets / classes (same
#'   dimension as the predictions that are passed as first argument).
#' @param feval Customized evaluation function. Just like `obj`, should take two arguments, with
#'   the first one being the predictions and the second one the `data` DMatrix.
#'
#'   Should return a list with two elements `metric` (name that will be displayed for this metric,
#'   should be a string / character), and `value` (the number that the function calculates, should
#'   be a numeric scalar).
#'
#'   Note that even if passing `feval`, objectives also have an associated default metric that
#'   will be evaluated in addition to it. In order to disable the built-in metric, one can pass
#'   parameter `disable_default_eval_metric = TRUE`.
#' @param verbose If 0, xgboost will stay silent. If 1, it will print information about performance.
#'   If 2, some additional information will be printed out.
#'   Note that setting `verbose > 0` automatically engages the
#'   `xgb.cb.print.evaluation(period=1)` callback function.
#' @param print_every_n Print each nth iteration evaluation messages when `verbose>0`.
#'   Default is 1 which means all messages are printed. This parameter is passed to the
#'   [xgb.cb.print.evaluation()] callback.
#' @param early_stopping_rounds If `NULL`, the early stopping function is not triggered.
#'   If set to an integer `k`, training with a validation set will stop if the performance
#'   doesn't improve for `k` rounds. Setting this parameter engages the [xgb.cb.early.stop()] callback.
#' @param maximize If `feval` and `early_stopping_rounds` are set, then this parameter must be set as well.
#'   When it is `TRUE`, it means the larger the evaluation score the better.
#'   This parameter is passed to the [xgb.cb.early.stop()] callback.
#' @param save_period When not `NULL`, model is saved to disk after every `save_period` rounds.
#'   0 means save at the end. The saving is handled by the [xgb.cb.save.model()] callback.
#' @param save_name the name or path for periodically saved model file.
#' @param xgb_model A previously built model to continue the training from.
#'   Could be either an object of class `xgb.Booster`, or its raw data, or the name of a
#'   file with a previously saved model.
#' @param callbacks A list of callback functions to perform various task during boosting.
#'   See [xgb.Callback()]. Some of the callbacks are automatically created depending on the
#'   parameters' values. User can provide either existing or their own callback methods in order
#'   to customize the training process.
#'
#'   Note that some callbacks might try to leave attributes in the resulting model object,
#'   such as an evaluation log (a `data.table` object) - be aware that these objects are kept
#'   as R attributes, and thus do not get saved when using XGBoost's own serializaters like
#'   [xgb.save()] (but are kept when using R serializers like [saveRDS()]).
#' @param ... other parameters to pass to `params`.
#'
#' @return An object of class `xgb.Booster`.
#'
#' @details
#' These are the training functions for [xgboost()].
#'
#' The `xgb.train()` interface supports advanced features such as `evals`,
#' customized objective and evaluation metric functions, therefore it is more flexible
#' than the [xgboost()] interface.
#'
#' Parallelization is automatically enabled if OpenMP is present.
#' Number of threads can also be manually specified via the `nthread` parameter.
#'
#' While in other interfaces, the default random seed defaults to zero, in R, if a parameter `seed`
#' is not manually supplied, it will generate a random seed through R's own random number generator,
#' whose seed in turn is controllable through `set.seed`. If `seed` is passed, it will override the
#' RNG from R.
#'
#' The evaluation metric is chosen automatically by XGBoost (according to the objective)
#' when the `eval_metric` parameter is not provided.
#' User may set one or several `eval_metric` parameters.
#' Note that when using a customized metric, only this single metric can be used.
#' The following is the list of built-in metrics for which XGBoost provides optimized implementation:
#' - `rmse`: Root mean square error. \url{https://en.wikipedia.org/wiki/Root_mean_square_error}
#' - `logloss`: Negative log-likelihood. \url{https://en.wikipedia.org/wiki/Log-likelihood}
#' - `mlogloss`: Multiclass logloss. \url{https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html}
#' - `error`: Binary classification error rate. It is calculated as `(# wrong cases) / (# all cases)`.
#'     By default, it uses the 0.5 threshold for predicted values to define negative and positive instances.
#'     Different threshold (e.g., 0.) could be specified as `error@0`.
#' - `merror`: Multiclass classification error rate. It is calculated as `(# wrong cases) / (# all cases)`.
#' - `mae`: Mean absolute error.
#' - `mape`: Mean absolute percentage error.
#' - `auc`: Area under the curve.
#'   \url{https://en.wikipedia.org/wiki/Receiver_operating_characteristic#'Area_under_curve} for ranking evaluation.
#' - `aucpr`: Area under the PR curve. \url{https://en.wikipedia.org/wiki/Precision_and_recall} for ranking evaluation.
#' - `ndcg`: Normalized Discounted Cumulative Gain (for ranking task). \url{https://en.wikipedia.org/wiki/NDCG}
#'
#' The following callbacks are automatically created when certain parameters are set:
#' - [xgb.cb.print.evaluation()] is turned on when `verbose > 0` and the `print_every_n`
#'   parameter is passed to it.
#' - [xgb.cb.evaluation.log()] is on when `evals` is present.
#' - [xgb.cb.early.stop()]: When `early_stopping_rounds` is set.
#' - [xgb.cb.save.model()]: When `save_period > 0` is set.
#'
#' Note that objects of type `xgb.Booster` as returned by this function behave a bit differently
#' from typical R objects (it's an 'altrep' list class), and it makes a separation between
#' internal booster attributes (restricted to jsonifyable data), accessed through [xgb.attr()]
#' and shared between interfaces through serialization functions like [xgb.save()]; and
#' R-specific attributes (typically the result from a callback), accessed through [attributes()]
#' and [attr()], which are otherwise
#' only used in the R interface, only kept when using R's serializers like [saveRDS()], and
#' not anyhow used by functions like `predict.xgb.Booster()`.
#'
#' Be aware that one such R attribute that is automatically added is `params` - this attribute
#' is assigned from the `params` argument to this function, and is only meant to serve as a
#' reference for what went into the booster, but is not used in other methods that take a booster
#' object - so for example, changing the booster's configuration requires calling `xgb.config<-`
#' or `xgb.parameters<-`, while simply modifying `attributes(model)$params$<...>` will have no
#' effect elsewhere.
#'
#' @seealso [xgb.Callback()], [predict.xgb.Booster()], [xgb.cv()]
#'
#' @references
#' Tianqi Chen and Carlos Guestrin, "XGBoost: A Scalable Tree Boosting System",
#' 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016, \url{https://arxiv.org/abs/1603.02754}
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#' data(agaricus.test, package = "xgboost")
#'
#' ## Keep the number of threads to 1 for examples
#' nthread <- 1
#' data.table::setDTthreads(nthread)
#'
#' dtrain <- with(
#'   agaricus.train, xgb.DMatrix(data, label = label, nthread = nthread)
#' )
#' dtest <- with(
#'   agaricus.test, xgb.DMatrix(data, label = label, nthread = nthread)
#' )
#' evals <- list(train = dtrain, eval = dtest)
#'
#' ## A simple xgb.train example:
#' param <- list(
#'   max_depth = 2,
#'   eta = 1,
#'   nthread = nthread,
#'   objective = "binary:logistic",
#'   eval_metric = "auc"
#' )
#' bst <- xgb.train(param, dtrain, nrounds = 2, evals = evals, verbose = 0)
#'
#' ## An xgb.train example where custom objective and evaluation metric are
#' ## used:
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
#' param <- list(
#'   max_depth = 2,
#'   eta = 1,
#'   nthread = nthread,
#'   objective = logregobj,
#'   eval_metric = evalerror
#' )
#' bst <- xgb.train(param, dtrain, nrounds = 2, evals = evals, verbose = 0)
#'
#' #  or through the ... arguments:
#' param <- list(max_depth = 2, eta = 1, nthread = nthread)
#' bst <- xgb.train(
#'   param,
#'   dtrain,
#'   nrounds = 2,
#'   evals = evals,
#'   verbose = 0,
#'   objective = logregobj,
#'   eval_metric = evalerror
#' )
#'
#' #  or as dedicated 'obj' and 'feval' parameters of xgb.train:
#' bst <- xgb.train(
#'   param, dtrain, nrounds = 2, evals = evals, obj = logregobj, feval = evalerror
#' )
#'
#'
#' ## An xgb.train example of using variable learning rates at each iteration:
#' param <- list(
#'   max_depth = 2,
#'   eta = 1,
#'   nthread = nthread,
#'   objective = "binary:logistic",
#'   eval_metric = "auc"
#' )
#' my_etas <- list(eta = c(0.5, 0.1))
#'
#' bst <- xgb.train(
#'  param,
#'  dtrain,
#'  nrounds = 2,
#'  evals = evals,
#'  verbose = 0,
#'  callbacks = list(xgb.cb.reset.parameters(my_etas))
#' )
#'
#' ## Early stopping:
#' bst <- xgb.train(
#'   param, dtrain, nrounds = 25, evals = evals, early_stopping_rounds = 3
#' )
#'
#' ## An 'xgboost' interface example:
#' bst <- xgboost(
#'   x = agaricus.train$data,
#'   y = factor(agaricus.train$label),
#'   params = list(max_depth = 2, eta = 1),
#'   nthread = nthread,
#'   nrounds = 2
#' )
#' pred <- predict(bst, agaricus.test$data)
#'
#' @export
xgb.train <- function(params = list(), data, nrounds, evals = list(),
                      obj = NULL, feval = NULL, verbose = 1, print_every_n = 1L,
                      early_stopping_rounds = NULL, maximize = NULL,
                      save_period = NULL, save_name = "xgboost.model",
                      xgb_model = NULL, callbacks = list(), ...) {

  check.deprecation(...)

  params <- check.booster.params(params, ...)

  check.custom.obj()
  check.custom.eval()

  # data & evals checks
  dtrain <- data
  if (!inherits(dtrain, "xgb.DMatrix"))
    stop("second argument dtrain must be xgb.DMatrix")
  if (length(evals) > 0) {
    if (typeof(evals) != "list" ||
        !all(vapply(evals, inherits, logical(1), what = 'xgb.DMatrix')))
      stop("'evals' must be a list of xgb.DMatrix elements")
    evnames <- names(evals)
    if (is.null(evnames) || any(evnames == ""))
      stop("each element of 'evals' must have a name tag")
  }
  # Handle multiple evaluation metrics given as a list
  for (m in params$eval_metric) {
    params <- c(params, list(eval_metric = m))
  }

  params <- c(params)
  params['validate_parameters'] <- TRUE
  if (!("seed" %in% names(params))) {
    params[["seed"]] <- sample(.Machine$integer.max, size = 1)
  }

  # callbacks
  tmp <- .process.callbacks(callbacks, is_cv = FALSE)
  callbacks <- tmp$callbacks
  cb_names <- tmp$cb_names
  rm(tmp)

  # Early stopping callback (should always come first)
  if (!is.null(early_stopping_rounds) && !("early_stop" %in% cb_names)) {
    callbacks <- add.callback(
      callbacks,
      xgb.cb.early.stop(
        early_stopping_rounds,
        maximize = maximize,
        verbose = verbose
      ),
      as_first_elt = TRUE
    )
  }
  # evaluation printing callback
  print_every_n <- max(as.integer(print_every_n), 1L)
  if (verbose && !("print_evaluation" %in% cb_names)) {
    callbacks <- add.callback(callbacks, xgb.cb.print.evaluation(print_every_n))
  }
  # evaluation log callback:  it is automatically enabled when 'evals' is provided
  if (length(evals) && !("evaluation_log" %in% cb_names)) {
    callbacks <- add.callback(callbacks, xgb.cb.evaluation.log())
  }
  # Model saving callback
  if (!is.null(save_period) && !("save_model" %in% cb_names)) {
    callbacks <- add.callback(callbacks, xgb.cb.save.model(save_period, save_name))
  }

  # The tree updating process would need slightly different handling
  is_update <- NVL(params[['process_type']], '.') == 'update'

  # Construct a booster (either a new one or load from xgb_model)
  bst <- xgb.Booster(
    params = params,
    cachelist = append(evals, dtrain),
    modelfile = xgb_model
  )
  niter_init <- bst$niter
  bst <- bst$bst
  .Call(
    XGBoosterCopyInfoFromDMatrix_R,
    xgb.get.handle(bst),
    dtrain
  )

  if (is_update && nrounds > niter_init)
    stop("nrounds cannot be larger than ", niter_init, " (nrounds of xgb_model)")

  niter_skip <- ifelse(is_update, 0, niter_init)
  begin_iteration <- niter_skip + 1
  end_iteration <- niter_skip + nrounds

  .execute.cb.before.training(
    callbacks,
    bst,
    dtrain,
    evals,
    begin_iteration,
    end_iteration
  )

  # the main loop for boosting iterations
  for (iteration in begin_iteration:end_iteration) {

    .execute.cb.before.iter(
      callbacks,
      bst,
      dtrain,
      evals,
      iteration
    )

    xgb.iter.update(
      bst = bst,
      dtrain = dtrain,
      iter = iteration - 1,
      obj = obj
    )

    bst_evaluation <- NULL
    if (length(evals) > 0) {
      bst_evaluation <- xgb.iter.eval(
        bst = bst,
        evals = evals,
        iter = iteration - 1,
        feval = feval
      )
    }

    should_stop <- .execute.cb.after.iter(
      callbacks,
      bst,
      dtrain,
      evals,
      iteration,
      bst_evaluation
    )

    if (should_stop) break
  }

  cb_outputs <- .execute.cb.after.training(
    callbacks,
    bst,
    dtrain,
    evals,
    iteration,
    bst_evaluation
  )

  extra_attrs <- list(
    call = match.call(),
    params = params
  )

  curr_attrs <- attributes(bst)
  if (NROW(curr_attrs)) {
    curr_attrs <- curr_attrs[
      setdiff(
        names(curr_attrs),
        c(names(extra_attrs), names(cb_outputs))
      )
    ]
  }
  curr_attrs <- c(extra_attrs, curr_attrs)
  if (NROW(cb_outputs)) {
    curr_attrs <- c(curr_attrs, cb_outputs)
  }
  attributes(bst) <- curr_attrs

  return(bst)
}
