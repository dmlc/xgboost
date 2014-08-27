# Main function for xgboost-package

xgboost = function(x=NULL,y=NULL,DMatrix=NULL, file=NULL, validation=NULL, 
                   nrounds=10, obj=NULL, feval=NULL, margin=NULL, verbose = T, ...)
{
    if (!is.null(DMatrix))
        dtrain = DMatrix
    else
    {
        if (is.null(x) && is.null(y))
        {
            if (is.null(file))
                stop('xgboost need input data, either R objects, local files or DMatrix object.')
            dtrain = xgb.DMatrix(file)
        }
        else
            dtrain = xgb.DMatrix(x, label=y)
        if (!is.null(margin))
        {
            succ <- xgb.setinfo(dtrain, "base_margin", margin)
            if (!succ)
                warning('Attemp to use margin failed.')
        }
    }
    
    params = list(...)
    
    watchlist=list()
    if (verbose)
    {
        if (!is.null(validation))
        {
            if (class(validation)!='xgb.DMatrix')
                dtest = xgb.DMatrix(validation)
            else
                dtest = validation
            watchlist = list(eval=dtest,train=dtrain)
        }
            
        else
            watchlist = list(train=dtrain)
    }
    
    bst <- xgb.train(params, dtrain, nrounds, watchlist, obj, feval)
    
    return(bst)
}


