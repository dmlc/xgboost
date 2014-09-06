require(xgboost)
require(methods)
# Directly read in local file
dtrain <- xgb.DMatrix("agaricus.txt.train")

history <- xgb.cv( data = dtrain, nround=3, nfold = 5, metrics=list("rmse","auc"),
                  "max_depth"=3, "eta"=1,
                  "objective"="binary:logistic")


