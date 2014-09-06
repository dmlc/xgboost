require(xgboost)
require(methods)
# Directly read in local file
dtrain <- xgb.DMatrix("agaricus.txt.train")

history <- xgb.cv(list("max_depth"=3, "eta"=1,
                       "objective"="binary:logistic"),
                  dtrain, nround=3, nfold = 5, "eval_metric"="error")

