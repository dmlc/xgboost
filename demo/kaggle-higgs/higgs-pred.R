# install xgboost package, see R-package in root folder
require(xgboost)
require(methods)

modelfile <- "higgs.model"
outfile <- "higgs.pred.csv"
dtest <- read.csv("data/test.csv", header=TRUE)
data <- as.matrix(dtest[2:31])
idx <- dtest[[1]]

xgmat <- xgb.DMatrix(data, missing = -999.0)
bst <- xgb.load(modelfile=modelfile)
ypred <- predict(bst, xgmat)

rorder <- rank(ypred, ties.method="first")

threshold <- 0.15
# to be completed
ntop <- length(rorder) - as.integer(threshold*length(rorder))
plabel <- ifelse(rorder > ntop, "s", "b")
outdata <- list("EventId" = idx,
                "RankOrder" = rorder,
                "Class" = plabel)
write.csv(outdata, file = outfile, quote=FALSE, row.names=FALSE)
