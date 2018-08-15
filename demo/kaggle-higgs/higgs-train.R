# install xgboost package, see R-package in root folder
require(xgboost)
require(methods)

testsize <- 550000

dtrain <- read.csv("data/training.csv", header=TRUE)
dtrain[33] <- dtrain[33] == "s"
label <- as.numeric(dtrain[[33]])
data <- as.matrix(dtrain[2:31])
weight <- as.numeric(dtrain[[32]]) * testsize / length(label)

sumwpos <- sum(weight * (label==1.0))
sumwneg <- sum(weight * (label==0.0))
print(paste("weight statistics: wpos=", sumwpos, "wneg=", sumwneg, "ratio=", sumwneg / sumwpos))

xgmat <- xgb.DMatrix(data, label = label, weight = weight, missing = -999.0)
param <- list("objective" = "binary:logitraw",
              "scale_pos_weight" = sumwneg / sumwpos,
              "bst:eta" = 0.1,
              "bst:max_depth" = 6,
              "eval_metric" = "auc",
              "eval_metric" = "ams@0.15",
              "silent" = 1,
              "nthread" = 16)
watchlist <- list("train" = xgmat)
nrounds = 120
print ("loading data end, start to boost trees")
bst = xgb.train(param, xgmat, nrounds, watchlist );
# save out model
xgb.save(bst, "higgs.model")
print ('finish training')

