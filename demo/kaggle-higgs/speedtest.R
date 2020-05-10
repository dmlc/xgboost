# install xgboost package, see R-package in root folder
require(xgboost)
require(gbm)
require(methods)

testsize <- 550000

dtrain <- read.csv("data/training.csv", header=TRUE, nrows=350001)
dtrain$Label = as.numeric(dtrain$Label=='s')
# gbm.time = system.time({
#   gbm.model <- gbm(Label ~ ., data = dtrain[, -c(1,32)], n.trees = 120, 
#                    interaction.depth = 6, shrinkage = 0.1, bag.fraction = 1,
#                    verbose = TRUE)
# })
# print(gbm.time)
# Test result: 761.48 secs

# dtrain[33] <- dtrain[33] == "s"
# label <- as.numeric(dtrain[[33]])
data <- as.matrix(dtrain[2:31])
weight <- as.numeric(dtrain[[32]]) * testsize / length(label)

sumwpos <- sum(weight * (label==1.0))
sumwneg <- sum(weight * (label==0.0))
print(paste("weight statistics: wpos=", sumwpos, "wneg=", sumwneg, "ratio=", sumwneg / sumwpos))

xgboost.time = list()
threads = c(1,2,4,8,16)
for (i in 1:length(threads)){
  thread = threads[i]
  xgboost.time[[i]] = system.time({
    xgmat <- xgb.DMatrix(data, label = label, weight = weight, missing = -999.0)
    param <- list("objective" = "binary:logitraw",
                  "scale_pos_weight" = sumwneg / sumwpos,
                  "bst:eta" = 0.1,
                  "bst:max_depth" = 6,
                  "eval_metric" = "auc",
                  "eval_metric" = "ams@0.15",
                  "silent" = 1,
                  "nthread" = thread)
    watchlist <- list("train" = xgmat)
    nrounds = 120
    print ("loading data end, start to boost trees")
    bst = xgb.train(param, xgmat, nrounds, watchlist );
    # save out model
    xgb.save(bst, "higgs.model")
    print ('finish training')
  })
}

xgboost.time
# [[1]]
# user  system elapsed 
# 99.015   0.051  98.982 
# 
# [[2]]
# user  system elapsed 
# 100.268   0.317  55.473 
# 
# [[3]]
# user  system elapsed 
# 111.682   0.777  35.963 
# 
# [[4]]
# user  system elapsed 
# 149.396   1.851  32.661 
# 
# [[5]]
# user  system elapsed 
# 157.390   5.988  40.949 

