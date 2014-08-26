# Main function for xgboost-package

xgboost = function(x=NULL,y=NULL,file=NULL,nrounds=10,params,watchlist=list(), 
                   obj=NULL, feval=NULL, margin=NULL)
{
    if (is.null(x) && is.null(y))
    {
        if (is.null(file))
            stop('xgboost need input data, either R objects or local files.')
        dtrain = xgb.DMatrix(file)
    }
    else
        dtrain = xgb.DMatrix(x, info=list(label=y))
    if (!is.null(margin))
    {
        succ <- xgb.setinfo(dtrain, "base_margin", margin)
        if (!succ)
            warning('Attemp to use margin failed.')
    }
    bst <- xgb.train(params, dtrain, nrounds, watchlist, obj, feval)
    return(bst)
}


