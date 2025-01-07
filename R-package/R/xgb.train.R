#' eXtreme Gradient Boosting Training
#'
#' `xgb.train()` is an advanced interface for training an xgboost model.
#' The [xgboost()] function is a simpler wrapper for `xgb.train()`.
#'
#' @param params List of XGBoost parameters which control the model building process.
#' See the [online documentation](http://xgboost.readthedocs.io/en/latest/parameter.html)
#' and the documentation for [xgb.params()] for details.
#'
#' Should be passed as list with named entries. Parameters that are not specified in this
#' list will use their default values.
#'
#' A list of named parameters can be created through the function [xgb.params()], which
#' accepts all valid parameters as function arguments.
#' @param data Training dataset. `xgb.train()` accepts only an `xgb.DMatrix` as the input.
#'
#' Note that there is a function [xgboost()] which is meant to accept R data objects
#' as inputs, such as data frames and matrices.
#' @param nrounds Max number of boosting iterations.
#' @param evals Named list of `xgb.DMatrix` datasets to use for evaluating model performance.
#'   Metrics specified in either `eval_metric` (under params) or `custom_metric` (function
#'   argument here) will be computed for each of these datasets during each boosting iteration,
#'   and stored in the end as a field named `evaluation_log` in the resulting object.
#'
#'   When either `verbose>=1` or [xgb.cb.print.evaluation()] callback is engaged, the performance
#'   results are continuously printed out during the training.
#'
#'   E.g., specifying `evals=list(validation1=mat1, validation2=mat2)` allows to track
#'   the performance of each round's model on `mat1` and `mat2`.
#' @param objective Customized objective function. Should take two arguments: the first one will be the
#'   current predictions (either a numeric vector or matrix depending on the number of targets / classes),
#'   and the second one will be the `data` DMatrix object that is used for training.
#'
#'   It should return a list with two elements `grad` and `hess` (in that order), as either
#'   numeric vectors or numeric matrices depending on the number of targets / classes (same
#'   dimension as the predictions that are passed as first argument).
#' @param custom_metric Customized evaluation function. Just like `objective`, should take two arguments,
#'   with the first one being the predictions and the second one the `data` DMatrix.
#'
#'   Should return a list with two elements `metric` (name that will be displayed for this metric,
#'   should be a string / character), and `value` (the number that the function calculates, should
#'   be a numeric scalar).
#'
#'   Note that even if passing `custom_metric`, objectives also have an associated default metric that
#'   will be evaluated in addition to it. In order to disable the built-in metric, one can pass
#'   parameter `disable_default_eval_metric = TRUE`.
#' @param verbose If 0, xgboost will stay silent. If 1, it will print information about performance.
#'   If 2, some additional information will be printed out.
#'   Note that setting `verbose > 0` automatically engages the
#'   `xgb.cb.print.evaluation(period=1)` callback function.
#' @param print_every_n When passing `verbose>0`, evaluation logs (metrics calculated on the
#' data passed under `evals`) will be printed every nth iteration according to the value passed
#' here. The first and last iteration are always included regardless of this 'n'.
#'
#' Only has an effect when passing data under `evals` and when passing `verbose>0`. The parameter
#' is passed to the [xgb.cb.print.evaluation()] callback.
#' @param early_stopping_rounds Number of boosting rounds after which training will be stopped
#'   if there is no improvement in performance (as measured by the evaluatiation metric that is
#'   supplied or selected by default for the objective) on the evaluation data passed under
#'   `evals`.
#'
#'   Must pass `evals` in order to use this functionality. Setting this parameter adds the
#'   [xgb.cb.early.stop()] callback.
#'
#'   If `NULL`, early stopping will not be used.
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
#' @param ... Not used.
#'
#' Some arguments that were part of this function in previous XGBoost versions are currently
#' deprecated or have been renamed. If a deprecated or renamed argument is passed, will throw
#' a warning (by default) and use its current equivalent instead. This warning will become an
#' error if using the \link[=xgboost-options]{'strict mode' option}.
#'
#' If some additional argument is passed that is neither a current function argument nor
#' a deprecated or renamed argument, a warning or error will be thrown depending on the
#' 'strict mode' option.
#'
#' \bold{Important:} `...` will be removed in a future version, and all the current
#' deprecation warnings will become errors. Please use only arguments that form part of
#' the function signature.
#' @return An object of class `xgb.Booster`.
#' @details
#' Compared to [xgboost()], the `xgb.train()` interface supports advanced features such as
#' `evals`, customized objective and evaluation metric functions, among others, with the
#' difference these work `xgb.DMatrix` objects and do not follow typical R idioms.
#'
#' Parallelization is automatically enabled if OpenMP is present.
#' Number of threads can also be manually specified via the `nthread` parameter.
#'
#' While in XGBoost language bindings, the default random seed defaults to zero, in R, if a parameter `seed`
#' is not manually supplied, it will generate a random seed through R's own random number generator,
#' whose seed in turn is controllable through `set.seed`. If `seed` is passed, it will override the
#' RNG from R.
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
#' or `xgb.model.parameters<-`, while simply modifying `attributes(model)$params$<...>` will have no
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
#' param <- xgb.params(
#'   max_depth = 2,
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
#' # These functions could be used by passing them as 'objective' and
#' # 'eval_metric' parameters in the params list:
#' param <- xgb.params(
#'   max_depth = 2,
#'   nthread = nthread,
#'   objective = logregobj,
#'   eval_metric = evalerror
#' )
#' bst <- xgb.train(param, dtrain, nrounds = 2, evals = evals, verbose = 0)
#'
#' # ... or as dedicated 'objective' and 'custom_metric' parameters of xgb.train:
#' bst <- xgb.train(
#'   within(param, rm("objective", "eval_metric")),
#'   dtrain, nrounds = 2, evals = evals,
#'   objective = logregobj, custom_metric = evalerror
#' )
#'
#'
#' ## An xgb.train example of using variable learning rates at each iteration:
#' param <- xgb.params(
#'   max_depth = 2,
#'   learning_rate = 1,
#'   nthread = nthread,
#'   objective = "binary:logistic",
#'   eval_metric = "auc"
#' )
#' my_learning_rates <- list(learning_rate = c(0.5, 0.1))
#'
#' bst <- xgb.train(
#'  param,
#'  dtrain,
#'  nrounds = 2,
#'  evals = evals,
#'  verbose = 0,
#'  callbacks = list(xgb.cb.reset.parameters(my_learning_rates))
#' )
#'
#' ## Early stopping:
#' bst <- xgb.train(
#'   param, dtrain, nrounds = 25, evals = evals, early_stopping_rounds = 3
#' )
#' @export
xgb.train <- function(params = xgb.params(), data, nrounds, evals = list(),
                      objective = NULL, custom_metric = NULL, verbose = 1, print_every_n = 1L,
                      early_stopping_rounds = NULL, maximize = NULL,
                      save_period = NULL, save_name = "xgboost.model",
                      xgb_model = NULL, callbacks = list(), ...) {
  check.deprecation(deprecated_train_params, match.call(), ...)

  params <- check.booster.params(params)
  tmp <- check.custom.obj(params, objective)
  params <- tmp$params
  objective <- tmp$objective
  tmp <- check.custom.eval(params, custom_metric, maximize, early_stopping_rounds, callbacks)
  params <- tmp$params
  custom_metric <- tmp$custom_metric

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
      objective = objective
    )

    bst_evaluation <- NULL
    if (length(evals) > 0) {
      bst_evaluation <- xgb.iter.eval(
        bst = bst,
        evals = evals,
        iter = iteration - 1,
        custom_metric = custom_metric
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

# nolint start: line_length_linter.
#' @title XGBoost Parameters
#' @description Convenience function to generate a list of named XGBoost parameters, which
#' can be passed as argument `params` to [xgb.train()]. See the [online documentation](
#' https://xgboost.readthedocs.io/en/stable/parameter.html) for more details.
#'
#' The purpose of this function is to enable IDE autocompletions and to provide in-package
#' documentation for all the possible parameters that XGBoost accepts. The output from this
#' function is just a regular R list containing the parameters that were set to non-default
#' values. Note that this function will not perform any validation on the supplied arguments.
#'
#' If passing `NULL` for a given parameter (the default for all of them), then the default
#' value for that parameter will be used. Default values are automatically determined by the
#' XGBoost core library upon calls to [xgb.train()] or [xgb.cv()], and are subject to change
#' over XGBoost library versions. Some of them might differ according to the
#' booster type (e.g. defaults for regularization are different for linear and tree-based boosters).
#' @return A list with the entries that were passed non-NULL values. It is intended to
#' be passed as argument `params` to [xgb.train()] or [xgb.cv()].
#' @export
#' @param objective (default=`"reg:squarederror"`)
#' Specify the learning task and the corresponding learning objective or a custom objective function to be used.
#'
#' For custom objective, see [Custom Objective and Evaluation Metric](https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html)
#' and [Custom objective and metric](https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html#custom-obj-metric) for more information,
#' along with the end note for function signatures.
#'
#' Supported values are:
#' - `"reg:squarederror"`: regression with squared loss.
#' - `"reg:squaredlogerror"`: regression with squared log loss \eqn{\frac{1}{2}[log(pred + 1) - log(label + 1)]^2}.  All input labels are required to be greater than -1.  Also, see metric `rmsle` for possible issue  with this objective.
#' - `"reg:logistic"`: logistic regression, output probability
#' - `"reg:pseudohubererror"`: regression with Pseudo Huber loss, a twice differentiable alternative to absolute loss.
#' - `"reg:absoluteerror"`: Regression with L1 error. When tree model is used, leaf value is refreshed after tree construction. If used in distributed training, the leaf value is calculated as the mean value from all workers, which is not guaranteed to be optimal.
#'
#'   Version added: 1.7.0
#' - `"reg:quantileerror"`: Quantile loss, also known as "pinball loss". See later sections for its parameter and [Quantile Regression](https://xgboost.readthedocs.io/en/latest/python/examples/quantile_regression.html#sphx-glr-python-examples-quantile-regression-py) for a worked example.
#'
#'   Version added: 2.0.0
#' - `"binary:logistic"`: logistic regression for binary classification, output probability
#' - `"binary:logitraw"`: logistic regression for binary classification, output score before logistic transformation
#' - `"binary:hinge"`: hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
#' - `"count:poisson"`: Poisson regression for count data, output mean of Poisson distribution.
#'   `"max_delta_step"` is set to 0.7 by default in Poisson regression (used to safeguard optimization)
#' - `"survival:cox"`: Cox regression for right censored survival time data (negative values are considered right censored).
#'
#'   Note that predictions are returned on the hazard ratio scale (i.e., as HR = exp(marginal_prediction) in the proportional hazard function `h(t) = h0(t) * HR`).
#' - `"survival:aft"`: Accelerated failure time model for censored survival time data.
#' See [Survival Analysis with Accelerated Failure Time](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html) for details.
#' - `"multi:softmax"`: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
#' - `"multi:softprob"`: same as softmax, but output a vector of `ndata * nclass`, which can be further reshaped to `ndata * nclass` matrix. The result contains predicted probability of each data point belonging to each class.
#' - `"rank:ndcg"`: Use LambdaMART to perform pair-wise ranking where [Normalized Discounted Cumulative Gain (NDCG)](http://en.wikipedia.org/wiki/NDCG) is maximized. This objective supports position debiasing for click data.
#' - `"rank:map"`: Use LambdaMART to perform pair-wise ranking where [Mean Average Precision (MAP)](http://en.wikipedia.org/wiki/Mean_average_precision#Mean_average_precision) is maximized
#' - `"rank:pairwise"`: Use LambdaRank to perform pair-wise ranking using the `ranknet` objective.
#' - `"reg:gamma"`: gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be [gamma-distributed](https://en.wikipedia.org/wiki/Gamma_distribution#Occurrence_and_applications).
#' - `"reg:tweedie"`: Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be [Tweedie-distributed](https://en.wikipedia.org/wiki/Tweedie_distribution#Occurrence_and_applications).
#' @param verbosity (default=1)
#' Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3
#' (debug). Sometimes XGBoost tries to change configurations based on heuristics, which
#' is displayed as warning message. If there's unexpected behaviour, please try to
#' increase value of verbosity.
#' @param nthread (default to maximum number of threads available if not set)
#' Number of parallel threads used to run XGBoost. When choosing it, please keep thread
#' contention and hyperthreading in mind.
#' @param seed Random number seed. If not specified, will take a random seed through R's own RNG engine.
#' @param booster (default= `"gbtree"`)
#' Which booster to use. Can be `"gbtree"`, `"gblinear"` or `"dart"`; `"gbtree"` and `"dart"` use tree based models while `"gblinear"` uses linear functions.
#' @param eta,learning_rate (two aliases for the same parameter)
#' Step size shrinkage used in update to prevent overfitting. After each boosting step, we can directly get the weights of new features, and `eta` shrinks the feature weights to make the boosting process more conservative.
#' - range: \eqn{[0,1]}
#' - default value: 0.3 for tree-based boosters, 0.5 for linear booster.
#'
#' Note: should only pass one of `eta` or `learning_rate`. Both refer to the same parameter and there's thus no difference between one or the other.
#' @param gamma,min_split_loss (two aliases for the same parameter) (for Tree Booster) (default=0, alias: `gamma`)
#' Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger `min_split_loss` is, the more conservative the algorithm will be. Note that a tree where no splits were made might still contain a single terminal node with a non-zero score.
#'
#' range: \eqn{[0, \infty)}
#'
#' Note: should only pass one of `gamma` or `min_split_loss`. Both refer to the same parameter and there's thus no difference between one or the other.
#' @param max_depth (for Tree Booster) (default=6)
#' Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree. `"exact"` tree method requires non-zero value.
#'
#' range: \eqn{[0, \infty)}
#' @param min_child_weight (for Tree Booster) (default=1)
#' Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than `min_child_weight`, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger `min_child_weight` is, the more conservative the algorithm will be.
#'
#' range: \eqn{[0, \infty)}
#' @param max_delta_step (for Tree Booster) (default=0)
#' Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
#'
#' range: \eqn{[0, \infty)}
#' @param subsample (for Tree Booster) (default=1)
#' Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
#'
#' range: \eqn{(0,1]}
#' @param sampling_method (for Tree Booster) (default= `"uniform"`)
#' The method to use to sample the training instances.
#' - `"uniform"`: each training instance has an equal probability of being selected. Typically set
#'   `"subsample"` >= 0.5 for good results.
#' - `"gradient_based"`: the selection probability for each training instance is proportional to the
#'   \bold{regularized absolute value} of gradients (more specifically, \eqn{\sqrt{g^2+\lambda h^2}}).
#'   `"subsample"` may be set to as low as 0.1 without loss of model accuracy. Note that this
#'   sampling method is only supported when `"tree_method"` is set to `"hist"` and the device is `"cuda"`; other tree
#'   methods only support `"uniform"` sampling.
#' @param colsample_bytree,colsample_bylevel,colsample_bynode (for Tree Booster) (default=1)
#' This is a family of parameters for subsampling of columns.
#' - All `"colsample_by*"` parameters have a range of \eqn{(0, 1]}, the default value of 1, and specify the fraction of columns to be subsampled.
#' - `"colsample_bytree"` is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed.
#' - `"colsample_bylevel"` is the subsample ratio of columns for each level. Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
#' - `"colsample_bynode"` is the subsample ratio of columns for each node (split). Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of columns chosen for the current level. This is not supported by the exact tree method.
#' - `"colsample_by*"` parameters work cumulatively. For instance,
#'   the combination `{'colsample_bytree'=0.5, 'colsample_bylevel'=0.5, 'colsample_bynode'=0.5}` with 64 features will leave 8 features to choose from at
#'   each split.
#'
#' One can set the `"feature_weights"` for DMatrix to
#' define the probability of each feature being selected when using column sampling.
#' @param lambda,reg_lambda (two aliases for the same parameter)
#'
#' - For tree-based boosters:
#'   - L2 regularization term on weights. Increasing this value will make model more conservative.
#'   - default: 1
#'   - range: \eqn{[0, \infty]}
#' - For linear booster:
#'   - L2 regularization term on weights. Increasing this value will make model more conservative. Normalised to number of training examples.
#'   - default: 0
#'   - range: \eqn{[0, \infty)}
#'
#' Note: should only pass one of `lambda` or `reg_lambda`. Both refer to the same parameter and there's thus no difference between one or the other.
#' @param alpha,reg_alpha (two aliases for the same parameter)
#' - L1 regularization term on weights. Increasing this value will make model more conservative.
#' - For the linear booster, it's normalised to number of training examples.
#' - default: 0
#' - range: \eqn{[0, \infty)}
#'
#' Note: should only pass one of `alpha` or `reg_alpha`. Both refer to the same parameter and there's thus no difference between one or the other.
#' @param tree_method (for Tree Booster) (default= `"auto"`)
#' The tree construction algorithm used in XGBoost. See description in the [reference paper](http://arxiv.org/abs/1603.02754) and [Tree Methods](https://xgboost.readthedocs.io/en/latest/treemethod.html).
#'
#' Choices: `"auto"`, `"exact"`, `"approx"`, `"hist"`, this is a combination of commonly
#' used updaters.  For other updaters like `"refresh"`, set the parameter `updater`
#' directly.
#' - `"auto"`: Same as the `"hist"` tree method.
#' - `"exact"`: Exact greedy algorithm.  Enumerates all split candidates.
#' - `"approx"`: Approximate greedy algorithm using quantile sketch and gradient histogram.
#' - `"hist"`: Faster histogram optimized approximate greedy algorithm.
#' @param scale_pos_weight (for Tree Booster) (default=1)
#' Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: `sum(negative instances) / sum(positive instances)`. See [Parameters Tuning](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html) for more discussion. Also, see Higgs Kaggle competition demo for examples: [R](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-train.R), [py1](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-numpy.py), [py2](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-cv.py), [py3](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py).
#' @param updater Has different meanings depending on the type of booster.
#'
#' - For tree-based boosters:
#'   A comma separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters. However, it could be also set explicitly by a user. The following updaters exist:
#'   - `"grow_colmaker"`: non-distributed column-based construction of trees.
#'   - `"grow_histmaker"`: distributed tree construction with row-based data splitting based on global proposal of histogram counting.
#'   - `"grow_quantile_histmaker"`: Grow tree using quantized histogram.
#'   - `"grow_gpu_hist"`:  Enabled when `tree_method` is set to `"hist"` along with `device="cuda"`.
#'   - `"grow_gpu_approx"`: Enabled when `tree_method` is set to `"approx"` along with `device="cuda"`.
#'   - `"sync"`: synchronizes trees in all distributed nodes.
#'   - `"refresh"`: refreshes tree's statistics and/or leaf values based on the current data. Note that no random subsampling of data rows is performed.
#'   - `"prune"`: prunes the splits where loss < `min_split_loss` (or `gamma`) and nodes that have depth greater than `max_depth`.
#'
#' - For `booster="gblinear"`:
#' (default= `"shotgun"`) Choice of algorithm to fit linear model
#'   - `"shotgun"`: Parallel coordinate descent algorithm based on shotgun algorithm. Uses 'hogwild' parallelism and therefore produces a nondeterministic solution on each run.
#'   - `"coord_descent"`: Ordinary coordinate descent algorithm. Also multithreaded but still produces a deterministic solution. When the `device` parameter is set to `"cuda"` or `"gpu"`, a GPU variant would be used.
#' @param refresh_leaf (for Tree Booster) (default=1)
#' This is a parameter of the `"refresh"` updater. When this flag is 1, tree leafs as well as tree nodes' stats are updated. When it is 0, only node stats are updated.
#' @param grow_policy (for Tree Booster) (default= `"depthwise"`)
#' - Controls a way new nodes are added to the tree.
#' - Currently supported only if `tree_method` is set to `"hist"` or `"approx"`.
#' - Choices: `"depthwise"`, `"lossguide"`
#'   - `"depthwise"`: split at nodes closest to the root.
#'   - `"lossguide"`: split at nodes with highest loss change.
#' @param max_leaves (for Tree Booster) (default=0)
#' Maximum number of nodes to be added.  Not used by `"exact"` tree method.
#' @param max_bin (for Tree Booster) (default=256)
#' - Only used if `tree_method` is set to `"hist"` or `"approx"`.
#' - Maximum number of discrete bins to bucket continuous features.
#' - Increasing this number improves the optimality of splits at the cost of higher computation time.
#' @param num_parallel_tree (for Tree Booster) (default=1)
#' Number of parallel trees constructed during each iteration. This option is used to support boosted random forest.
#' @param monotone_constraints (for Tree Booster)
#' Constraint of variable monotonicity. See [Monotonic Constraints](https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html) for more information.
#' @param interaction_constraints (for Tree Booster)
#' Constraints for interaction representing permitted interactions. The constraints must
#' be specified in the form of a nest list, e.g. `list(c(0, 1), c(2, 3, 4))`, where each inner
#' list is a group of indices of features (base-0 numeration) that are allowed to interact with each other.
#' See [Feature Interaction Constraints](https://xgboost.readthedocs.io/en/latest/tutorials/feature_interaction_constraint.html) for more information.
#' @param multi_strategy (for Tree Booster) (default = `"one_output_per_tree"`)
#' The strategy used for training multi-target models, including multi-target regression
#' and multi-class classification. See [Multiple Outputs](https://xgboost.readthedocs.io/en/latest/tutorials/multioutput.html) for more information.
#' - `"one_output_per_tree"`: One model for each target.
#' - `"multi_output_tree"`:  Use multi-target trees.
#'
#' Version added: 2.0.0
#'
#' Note: This parameter is working-in-progress.
#' @param base_score
#' - The initial prediction score of all instances, global bias
#' - The parameter is automatically estimated for selected objectives before training. To
#'   disable the estimation, specify a real number argument.
#' - If `base_margin` is supplied, `base_score` will not be added.
#' - For sufficient number of iterations, changing this value will not have too much effect.
#' @param eval_metric (default according to objective)
#' - Evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and logloss for classification, `mean average precision` for ``rank:map``, etc.)
#' - User can add multiple evaluation metrics.
#' - The choices are listed below:
#'   - `"rmse"`: [root mean square error](http://en.wikipedia.org/wiki/Root_mean_square_error)
#'   - `"rmsle"`: root mean square log error: \eqn{\sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}}. Default metric of `"reg:squaredlogerror"` objective. This metric reduces errors generated by outliers in dataset.  But because `log` function is employed, `"rmsle"` might output `nan` when prediction value is less than -1.  See `"reg:squaredlogerror"` for other requirements.
#'   - `"mae"`: [mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)
#'   - `"mape"`: [mean absolute percentage error](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)
#'   - `"mphe"`: [mean Pseudo Huber error](https://en.wikipedia.org/wiki/Huber_loss). Default metric of `"reg:pseudohubererror"` objective.
#'   - `"logloss"`: [negative log-likelihood](http://en.wikipedia.org/wiki/Log-likelihood)
#'   - `"error"`: Binary classification error rate. It is calculated as `#(wrong cases)/#(all cases)`. For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
#'   - `"error@t"`: a different than 0.5 binary classification threshold value could be specified by providing a numerical value through 't'.
#'   - `"merror"`: Multiclass classification error rate. It is calculated as `#(wrong cases)/#(all cases)`.
#'   - `"mlogloss"`: [Multiclass logloss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html).
#'   - `"auc"`: [Receiver Operating Characteristic Area under the Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve).
#'     Available for classification and learning-to-rank tasks.
#'     - When used with binary classification, the objective should be `"binary:logistic"` or similar functions that work on probability.
#'     - When used with multi-class classification, objective should be `"multi:softprob"` instead of `"multi:softmax"`, as the latter doesn't output probability.  Also the AUC is calculated by 1-vs-rest with reference class weighted by class prevalence.
#'     - When used with LTR task, the AUC is computed by comparing pairs of documents to count correctly sorted pairs.  This corresponds to pairwise learning to rank.  The implementation has some issues with average AUC around groups and distributed workers not being well-defined.
#'     - On a single machine the AUC calculation is exact. In a distributed environment the AUC is a weighted average over the AUC of training rows on each node - therefore, distributed AUC is an approximation sensitive to the distribution of data across workers. Use another metric in distributed environments if precision and reproducibility are important.
#'     - When input dataset contains only negative or positive samples, the output is `NaN`.  The behavior is implementation defined, for instance, `scikit-learn` returns \eqn{0.5} instead.
#'   - `"aucpr"`: [Area under the PR curve](https://en.wikipedia.org/wiki/Precision_and_recall).
#'     Available for classification and learning-to-rank tasks.
#'
#'     After XGBoost 1.6, both of the requirements and restrictions for using `"aucpr"` in classification problem are similar to `"auc"`.  For ranking task, only binary relevance label \eqn{y \in [0, 1]} is supported.  Different from `"map"` (mean average precision), `"aucpr"` calculates the *interpolated* area under precision recall curve using continuous interpolation.
#'
#'   - `"pre"`: Precision at \eqn{k}. Supports only learning to rank task.
#'   - `"ndcg"`: [Normalized Discounted Cumulative Gain](http://en.wikipedia.org/wiki/NDCG)
#'   - `"map"`: [Mean Average Precision](http://en.wikipedia.org/wiki/Mean_average_precision#Mean_average_precision)
#'
#'     The `average precision` is defined as:
#'
#'       \eqn{AP@l = \frac{1}{min{(l, N)}}\sum^l_{k=1}P@k \cdot I_{(k)}}
#'
#'     where \eqn{I_{(k)}} is an indicator function that equals to \eqn{1} when the document at \eqn{k} is relevant and \eqn{0} otherwise. The \eqn{P@k} is the precision at \eqn{k}, and \eqn{N} is the total number of relevant documents. Lastly, the `mean average precision` is defined as the weighted average across all queries.
#'
#'   - `"ndcg@n"`, `"map@n"`, `"pre@n"`: \eqn{n} can be assigned as an integer to cut off the top positions in the lists for evaluation.
#'   - `"ndcg-"`, `"map-"`, `"ndcg@n-"`, `"map@n-"`: In XGBoost, the NDCG and MAP evaluate the score of a list without any positive samples as \eqn{1}. By appending "-" to the evaluation metric name, we can ask XGBoost to evaluate these scores as \eqn{0} to be consistent under some conditions.
#'   - `"poisson-nloglik"`: negative log-likelihood for Poisson regression
#'   - `"gamma-nloglik"`: negative log-likelihood for gamma regression
#'   - `"cox-nloglik"`: negative partial log-likelihood for Cox proportional hazards regression
#'   - `"gamma-deviance"`: residual deviance for gamma regression
#'   - `"tweedie-nloglik"`: negative log-likelihood for Tweedie regression (at a specified value of the `tweedie_variance_power` parameter)
#'   - `"aft-nloglik"`: Negative log likelihood of Accelerated Failure Time model.
#'     See [Survival Analysis with Accelerated Failure Time](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html) for details.
#'   - `"interval-regression-accuracy"`: Fraction of data points whose predicted labels fall in the interval-censored labels.
#'     Only applicable for interval-censored data.  See [Survival Analysis with Accelerated Failure Time](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html) for details.
#' @param seed_per_iteration (default= `FALSE`)
#' Seed PRNG determnisticly via iterator number.
#' @param device (default= `"cpu"`)
#' Device for XGBoost to run. User can set it to one of the following values:
#' - `"cpu"`: Use CPU.
#' - `"cuda"`: Use a GPU (CUDA device).
#' - `"cuda:<ordinal>"`: `<ordinal>` is an integer that specifies the ordinal of the GPU (which GPU do you want to use if you have more than one devices).
#' - `"gpu"`: Default GPU device selection from the list of available and supported devices. Only `"cuda"` devices are supported currently.
#' - `"gpu:<ordinal>"`: Default GPU device selection from the list of available and supported devices. Only `"cuda"` devices are supported currently.
#'
#' For more information about GPU acceleration, see [XGBoost GPU Support](https://xgboost.readthedocs.io/en/latest/gpu/index.html). In distributed environments, ordinal selection is handled by distributed frameworks instead of XGBoost. As a result, using `"cuda:<ordinal>"` will result in an error. Use `"cuda"` instead.
#'
#' Version added: 2.0.0
#'
#' Note: if XGBoost was installed from CRAN, it won't have GPU support enabled, thus only `"cpu"` will be available.
#' To get GPU support, the R package for XGBoost must be installed from source or from the GitHub releases - see
#' [instructions](https://xgboost.readthedocs.io/en/latest/install.html#r).
#' @param disable_default_eval_metric (default= `FALSE`)
#' Flag to disable default metric. Set to 1 or `TRUE` to disable.
#' @param use_rmm Whether to use RAPIDS Memory Manager (RMM) to allocate cache GPU
#' memory. The primary memory is always allocated on the RMM pool when XGBoost is built
#' (compiled) with the RMM plugin enabled. Valid values are `TRUE` and `FALSE`. See
#' [Using XGBoost with RAPIDS Memory Manager (RMM) plugin](https://xgboost.readthedocs.io/en/latest/python/rmm-examples/index.html) for details.
#' @param max_cached_hist_node (for Non-Exact Tree Methods) (default = 65536)
#' Maximum number of cached nodes for histogram. This can be used with the `"hist"` and the
#' `"approx"` tree methods.
#'
#' Version added: 2.0.0
#'
#' - For most of the cases this parameter should not be set except for growing deep
#'   trees. After 3.0, this parameter affects GPU algorithms as well.
#' @param extmem_single_page (for Non-Exact Tree Methods) (default = `FALSE`)
#' This parameter is only used for the `"hist"` tree method with `device="cuda"` and
#' `subsample != 1.0`. Before 3.0, pages were always concatenated.
#'
#' Version added: 3.0.0
#'
#' Whether the GPU-based `"hist"` tree method should concatenate the training data into a
#' single batch instead of fetching data on-demand when external memory is used. For GPU
#' devices that don't support address translation services, external memory training is
#' expensive. This parameter can be used in combination with subsampling to reduce overall
#' memory usage without significant overhead. See [Using XGBoost External Memory Version](https://xgboost.readthedocs.io/en/latest/tutorials/external_memory.html) for
#' more information.
#' @param max_cat_to_onehot (for Non-Exact Tree Methods)
#' A threshold for deciding whether XGBoost should use one-hot encoding based split for
#' categorical data.  When number of categories is lesser than the threshold then one-hot
#' encoding is chosen, otherwise the categories will be partitioned into children nodes.
#'
#' Version added: 1.6.0
#' @param max_cat_threshold (for Non-Exact Tree Methods)
#' Maximum number of categories considered for each split. Used only by partition-based
#' splits for preventing over-fitting.
#'
#' Version added: 1.7.0
#' @param sample_type (for Dart Booster) (default= `"uniform"`)
#' Type of sampling algorithm.
#' - `"uniform"`: dropped trees are selected uniformly.
#' - `"weighted"`: dropped trees are selected in proportion to weight.
#' @param normalize_type (for Dart Booster) (default= `"tree"`)
#' Type of normalization algorithm.
#' - `"tree"`: new trees have the same weight of each of dropped trees.
#'   - Weight of new trees are `1 / (k + learning_rate)`.
#'   - Dropped trees are scaled by a factor of `k / (k + learning_rate)`.
#' - `"forest"`: new trees have the same weight of sum of dropped trees (forest).
#'   - Weight of new trees are `1 / (1 + learning_rate)`.
#'   - Dropped trees are scaled by a factor of `1 / (1 + learning_rate)`.
#' @param rate_drop (for Dart Booster) (default=0.0)
#' Dropout rate (a fraction of previous trees to drop during the dropout).
#'
#' range: \eqn{[0.0, 1.0]}
#' @param one_drop (for Dart Booster) (default=0)
#' When this flag is enabled, at least one tree is always dropped during the dropout (allows Binomial-plus-one or epsilon-dropout from the original DART paper).
#' @param skip_drop (for Dart Booster) (default=0.0)
#' Probability of skipping the dropout procedure during a boosting iteration.
#' - If a dropout is skipped, new trees are added in the same manner as `"gbtree"`.
#' - Note that non-zero `skip_drop` has higher priority than `rate_drop` or `one_drop`.
#'
#' range: \eqn{[0.0, 1.0]}
#' @param feature_selector (for Linear Booster) (default= `"cyclic"`)
#' Feature selection and ordering method
#' - `"cyclic"`: Deterministic selection by cycling through features one at a time.
#' - `"shuffle"`: Similar to `"cyclic"` but with random feature shuffling prior to each update.
#' - `"random"`: A random (with replacement) coordinate selector.
#' - `"greedy"`: Select coordinate with the greatest gradient magnitude.  It has `O(num_feature^2)` complexity. It is fully deterministic. It allows restricting the selection to `top_k` features per group with the largest magnitude of univariate weight change, by setting the `top_k` parameter. Doing so would reduce the complexity to `O(num_feature*top_k)`.
#' - `"thrifty"`: Thrifty, approximately-greedy feature selector. Prior to cyclic updates, reorders features in descending magnitude of their univariate weight changes. This operation is multithreaded and is a linear complexity approximation of the quadratic greedy selection. It allows restricting the selection to `top_k` features per group with the largest magnitude of univariate weight change, by setting the `top_k` parameter.
#' @param top_k (for Linear Booster) (default=0)
#' The number of top features to select in `greedy` and `thrifty` feature selector. The value of 0 means using all the features.
#' @param num_class Number of classes when using multi-class classification objectives (e.g. `objective="multi:softprob"`)
#' @param tweedie_variance_power (for Tweedie Regression (`"objective=reg:tweedie"`)) (default=1.5)
#' - Parameter that controls the variance of the Tweedie distribution `var(y) ~ E(y)^tweedie_variance_power`
#' - range: \eqn{(1,2)}
#' - Set closer to 2 to shift towards a gamma distribution
#' - Set closer to 1 to shift towards a Poisson distribution.
#' @param huber_slope (for using Pseudo-Huber (`"reg:pseudohubererror`")) (default = 1.0)
#' A parameter used for Pseudo-Huber loss to define the \eqn{\delta} term.
#' @param quantile_alpha (for using Quantile Loss (`"reg:quantileerror"`))
#' A scalar or a list of targeted quantiles (passed as a numeric vector).
#'
#' Version added: 2.0.0
#' @param aft_loss_distribution (for using AFT Survival Loss (`"survival:aft"`) and Negative Log Likelihood of AFT metric (`"aft-nloglik"`))
#' Probability Density Function, `"normal"`, `"logistic"`, or `"extreme"`.
#' @param lambdarank_pair_method (for learning to rank (`"rank:ndcg"`, `"rank:map"`, `"rank:pairwise"`)) (default = `"topk"`)
#' How to construct pairs for pair-wise learning.
#' - `"mean"`: Sample `lambdarank_num_pair_per_sample` pairs for each document in the query list.
#' - `"topk"`: Focus on top-`lambdarank_num_pair_per_sample` documents. Construct \eqn{|query|} pairs for each document at the top-`lambdarank_num_pair_per_sample` ranked by the model.
#' @param lambdarank_num_pair_per_sample (for learning to rank (`"rank:ndcg"`, `"rank:map"`, `"rank:pairwise"`))
#' It specifies the number of pairs sampled for each document when pair method is `"mean"`, or the truncation level for queries when the pair method is `"topk"`. For example, to train with `ndcg@6`, set `"lambdarank_num_pair_per_sample"` to \eqn{6} and `lambdarank_pair_method` to `"topk"`.
#'
#' range = \eqn{[1, \infty)}
#' @param lambdarank_normalization (for learning to rank (`"rank:ndcg"`, `"rank:map"`, `"rank:pairwise"`)) (default = `TRUE`)
#' Whether to normalize the leaf value by lambda gradient. This can sometimes stagnate the training progress.
#'
#' Version added: 2.1.0
#' @param lambdarank_unbiased (for learning to rank (`"rank:ndcg"`, `"rank:map"`, `"rank:pairwise"`)) (default = `FALSE`)
#' Specify whether do we need to debias input click data.
#' @param lambdarank_bias_norm (for learning to rank (`"rank:ndcg"`, `"rank:map"`, `"rank:pairwise"`)) (default = 2.0)
#' \eqn{L_p} normalization for position debiasing, default is \eqn{L_2}. Only relevant when `lambdarank_unbiased` is set to `TRUE`.
#' @param ndcg_exp_gain (for learning to rank (`"rank:ndcg"`, `"rank:map"`, `"rank:pairwise"`)) (default = `TRUE`)
#' Whether we should use exponential gain function for `NDCG`. There are two forms of gain function for `NDCG`, one is using relevance value directly while the other is using\eqn{2^{rel} - 1} to emphasize on retrieving relevant documents. When `ndcg_exp_gain` is `TRUE` (the default), relevance degree cannot be greater than 31.
xgb.params <- function(
  objective = NULL,
  verbosity = NULL,
  nthread = NULL,
  seed = NULL,
  booster = NULL,
  eta = NULL,
  learning_rate = NULL,
  gamma = NULL,
  min_split_loss = NULL,
  max_depth = NULL,
  min_child_weight = NULL,
  max_delta_step = NULL,
  subsample = NULL,
  sampling_method = NULL,
  colsample_bytree = NULL,
  colsample_bylevel = NULL,
  colsample_bynode = NULL,
  lambda = NULL,
  reg_lambda = NULL,
  alpha = NULL,
  reg_alpha = NULL,
  tree_method = NULL,
  scale_pos_weight = NULL,
  updater = NULL,
  refresh_leaf = NULL,
  grow_policy = NULL,
  max_leaves = NULL,
  max_bin = NULL,
  num_parallel_tree = NULL,
  monotone_constraints = NULL,
  interaction_constraints = NULL,
  multi_strategy = NULL,
  base_score = NULL,
  eval_metric = NULL,
  seed_per_iteration = NULL,
  device = NULL,
  disable_default_eval_metric = NULL,
  use_rmm = NULL,
  max_cached_hist_node = NULL,
  extmem_single_page = NULL,
  max_cat_to_onehot = NULL,
  max_cat_threshold = NULL,
  sample_type = NULL,
  normalize_type = NULL,
  rate_drop = NULL,
  one_drop = NULL,
  skip_drop = NULL,
  feature_selector = NULL,
  top_k = NULL,
  num_class = NULL,
  tweedie_variance_power = NULL,
  huber_slope = NULL,
  quantile_alpha = NULL,
  aft_loss_distribution = NULL,
  lambdarank_pair_method = NULL,
  lambdarank_num_pair_per_sample = NULL,
  lambdarank_normalization = NULL,
  lambdarank_unbiased = NULL,
  lambdarank_bias_norm = NULL,
  ndcg_exp_gain = NULL
) {
# nolint end
  out <- as.list(environment())
  out <- out[!sapply(out, is.null)]
  return(out)
}
