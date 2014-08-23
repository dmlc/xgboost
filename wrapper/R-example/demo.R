# include xgboost library, must set chdir=TRURE
source('../xgboost.R', chdir=TRUE)

# test code here
dtrain <- xgb.DMatrix("agaricus.txt.train")
dtest <- xgb.DMatrix("agaricus.txt.test")
param = list('bst:max_depth'=2, 'bst:eta'=1, 'silent'=1, 'objective'='binary:logistic')
watchlist <- list('train'=dtrain,'test'=dtest)
bst <- xgb.train(param, dtrain, watchlist=watchlist, nround=3)

succ <- xgb.save(bst, "iter.model")
print('finsih save model')
bst2 <- xgb.Booster(modelfile="iter.model")
pred = xgb.predict(bst2, dtest)
