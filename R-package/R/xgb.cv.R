#' Cross Validation
#' 
#' The cross valudation function of xgboost
#' 
#' @importFrom data.table data.table
#' @importFrom data.table as.data.table
#' @importFrom magrittr %>%
#' @importFrom data.table :=
#' @importFrom data.table rbindlist
#' @importFrom stringr str_extract_all
#' @importFrom stringr str_extract
#' @importFrom stringr str_split
#' @importFrom stringr str_replace
#' @importFrom stringr str_match
#' 
#' @param params the list of parameters. Commonly used ones are:
#' \itemize{
#'   \item \code{objective} objective function, common ones are
#'   \itemize{
#'     \item \code{reg:linear} linear regression
#'     \item \code{binary:logistic} logistic regression for classification
#'   }
#'   \item \code{eta} step size of each boosting step
#'   \item \code{max.depth} maximum depth of the tree
#'   \item \code{nthread} number of thread used in training, if not set, all threads are used
#' }
#'
#'   See \link{xgb.train} for further details.
#'   See also demo/ for walkthrough example in R.
#' @param data takes an \code{xgb.DMatrix} or \code{Matrix} as the input.
#' @param nrounds the max number of iterations
#' @param nfold the original dataset is randomly partitioned into \code{nfold} equal size subsamples. 
#' @param label option field, when data is \code{Matrix}
#' @param missing Missing is only used when input is dense matrix, pick a float
#'     value that represents missing value. Sometime a data use 0 or other extreme value to represents missing values.
#' @param prediction A logical value indicating whether to return the prediction vector.
#' @param showsd \code{boolean}, whether show standard deviation of cross validation
#' @param metrics, list of evaluation metrics to be used in corss validation,
#'   when it is not specified, the evaluation metric is chosen according to objective function.
#'   Possible options are:
#' \itemize{
#'   \item \code{error} binary classification error rate
#'   \item \code{rmse} Rooted mean square error
#'   \item \code{logloss} negative log-likelihood function
#'   \item \code{auc} Area under curve
#'   \item \code{merror} Exact matching error, used to evaluate multi-class classification
#' }
#' @param obj customized objective function. Returns gradient and second order 
#'   gradient with given prediction and dtrain.
#' @param feval custimized evaluation function. Returns 
#'   \code{list(metric='metric-name', value='metric-value')} with given 
#'   prediction and dtrain.
#' @param stratified \code{boolean} whether sampling of folds should be stratified by the values of labels in \code{data}
#' @param folds \code{list} provides a possibility of using a list of pre-defined CV folds (each element must be a vector of fold's indices).
#'   If folds are supplied, the nfold and stratified parameters would be ignored.
#' @param verbose \code{boolean}, print the statistics during the process
#' @param print.every.n Print every N progress messages when \code{verbose>0}. Default is 1 which means all messages are printed.
#' @param early.stop.round If \code{NULL}, the early stopping function is not triggered. 
#'     If set to an integer \code{k}, training with a validation set will stop if the performance 
#'     keeps getting worse consecutively for \code{k} rounds.
#' @param maximize If \code{feval} and \code{early.stop.round} are set, then \code{maximize} must be set as well.
#'     \code{maximize=TRUE} means the larger the evaluation score the better.
#'     
#' @param ... other parameters to pass to \code{params}.
#' 
#' @return
#' If \code{prediction = TRUE}, a list with the following elements is returned:
#' \itemize{
#'   \item \code{dt} a \code{data.table} with each mean and standard deviation stat for training set and test set
#'   \item \code{pred} an array or matrix (for multiclass classification) with predictions for each CV-fold for the model having been trained on the data in all other folds.
#' }
#'
#' If \code{prediction = FALSE}, just a \code{data.table} with each mean and standard deviation stat for training set and test set is returned.
#'
#' @details 
#' The original sample is randomly partitioned into \code{nfold} equal size subsamples. 
#' 
#' Of the \code{nfold} subsamples, a single subsample is retained as the validation data for testing the model, and the remaining \code{nfold - 1} subsamples are used as training data. 
#' 
#' The cross-validation process is then repeated \code{nrounds} times, with each of the \code{nfold} subsamples used exactly once as the validation data.
#' 
#' All observations are used for both training and validation.
#' 
#' Adapted from \url{http://en.wikipedia.org/wiki/Cross-validation_\%28statistics\%29#k-fold_cross-validation}
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
#' history <- xgb.cv(data = dtrain, nround=3, nthread = 2, nfold = 5, metrics=list("rmse","auc"),
#'                   max.depth =3, eta = 1, objective = "binary:logistic")
#' print(history)
#' @export
#'
xgb.cv <- function(params=list(), data, nrounds, nfold, label = NULL, missing = NULL, 
                   prediction = FALSE, showsd = TRUE, metrics=list(), 
                   obj = NULL, feval = NULL, stratified = TRUE, folds = NULL, verbose = T, print.every.n=1L,
                   early.stop.round = NULL, maximize = NULL, ...) {
    if (typeof(params) != "list") {
        stop("xgb.cv: first argument params must be list")
    }
    if(!is.null(folds)) {
        if(class(folds)!="list" | length(folds) < 2) {
            stop("folds must be a list with 2 or more elements that are vectors of indices for each CV-fold")
        }
        nfold <- length(folds)
    }
    if (nfold <= 1) {
        stop("nfold must be bigger than 1")
    }
    if (is.null(missing)) {
        dtrain <- xgb.get.DMatrix(data, label)
    } else {
        dtrain <- xgb.get.DMatrix(data, label, missing)
    }
    dot.params = list(...)
    nms.params = names(params)
    nms.dot.params = names(dot.params)
    if (length(intersect(nms.params,nms.dot.params))>0)
        stop("Duplicated defined term in parameters. Please check your list of params.")
    params <- append(params, dot.params)
    params <- append(params, list(silent=1))
    for (mc in metrics) {
        params <- append(params, list("eval_metric"=mc))
    }
    
    # customized objective and evaluation metric interface
    if (!is.null(params$objective) && !is.null(obj))
        stop("xgb.cv: cannot assign two different objectives")
    if (!is.null(params$objective))
        if (class(params$objective)=='function') {
            obj = params$objective
            params[['objective']] = NULL
        }
    # if (!is.null(params$eval_metric) && !is.null(feval))
    #  stop("xgb.cv: cannot assign two different evaluation metrics")
    if (!is.null(params$eval_metric))
        if (class(params$eval_metric)=='function') {
            feval = params$eval_metric
            params[['eval_metric']] = NULL
        }
    
    # Early Stopping
    if (!is.null(early.stop.round)){
        if (!is.null(feval) && is.null(maximize))
            stop('Please set maximize to note whether the model is maximizing the evaluation or not.')
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
        
        if (length(metrics)>1)
            warning('Only the first metric is used for early stopping process.')
    }
    
    xgb_folds <- xgb.cv.mknfold(dtrain, nfold, params, stratified, folds)
    obj_type = params[['objective']]
    mat_pred = FALSE
    if (!is.null(obj_type) && obj_type=='multi:softprob')
    {
        num_class = params[['num_class']]
        if (is.null(num_class))
            stop('must set num_class to use softmax')
        predictValues <- matrix(0,xgb.numrow(dtrain),num_class)
        mat_pred = TRUE
    }
    else
        predictValues <- rep(0,xgb.numrow(dtrain))
    history <- c()
    print.every.n = max(as.integer(print.every.n), 1L)
    for (i in 1:nrounds) {
        msg <- list()
        for (k in 1:nfold) {
            fd <- xgb_folds[[k]]
            succ <- xgb.iter.update(fd$booster, fd$dtrain, i - 1, obj)
            msg[[k]] <- xgb.iter.eval(fd$booster, fd$watchlist, i - 1, feval) %>% str_split("\t") %>% .[[1]]
        }
        ret <- xgb.cv.aggcv(msg, showsd)
        history <- c(history, ret)
        if(verbose)
            if (0==(i-1L)%%print.every.n)
                cat(ret, "\n", sep="")
        
        # early_Stopping
        if (!is.null(early.stop.round)){
            score = strsplit(ret,'\\s+')[[1]][1+length(metrics)+2]
            score = strsplit(score,'\\+|:')[[1]][[2]]
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
    
    if (prediction) {
        for (k in 1:nfold) {
            fd = xgb_folds[[k]]
            if (!is.null(early.stop.round) && earlyStopflag) {
              res = xgb.iter.eval(fd$booster, fd$watchlist, bestInd - 1, feval, prediction)
            } else {
              res = xgb.iter.eval(fd$booster, fd$watchlist, nrounds - 1, feval, prediction)
            }
            if (mat_pred) {
                pred_mat = matrix(res[[2]],num_class,length(fd$index))
                predictValues[fd$index,] = t(pred_mat)
            } else {
                predictValues[fd$index] = res[[2]]
            }
        }
    }
    
    
    colnames <- str_split(string = history[1], pattern = "\t")[[1]] %>% .[2:length(.)] %>% str_extract(".*:") %>% str_replace(":","") %>% str_replace("-", ".")
    colnamesMean <- paste(colnames, "mean")
    if(showsd) colnamesStd <- paste(colnames, "std")
    
    colnames <- c()
    if(showsd) for(i in 1:length(colnamesMean)) colnames <- c(colnames, colnamesMean[i], colnamesStd[i])
    else colnames <- colnamesMean
    
    type <- rep(x = "numeric", times = length(colnames))
    dt <- utils::read.table(text = "", colClasses = type, col.names = colnames) %>% as.data.table
    split <- str_split(string = history, pattern = "\t")
    
    for(line in split) dt <- line[2:length(line)] %>% str_extract_all(pattern = "\\d*\\.+\\d*") %>% unlist %>% as.numeric %>% as.list %>% {rbindlist(list(dt, .), use.names = F, fill = F)}
    
    if (prediction) {
        return(list(dt = dt,pred = predictValues))
    }
    return(dt)
}

# Avoid error messages during CRAN check.
# The reason is that these variables are never declared
# They are mainly column names inferred by Data.table...
globalVariables(".")
