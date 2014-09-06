require(xgboost)

dtrain <- xgb.DMatrix('../data/agaricus.txt.train')
dtest <- xgb.DMatrix('../data/agaricus.txt.test')
param <- list(max_depth=2,eta=1,silent=1,objective='binary:logistic')
watchlist <- list(eval = dtest, train = dtrain)
num_round <- 2
bst <- xgb.train(param, dtrain, num_round, watchlist)
preds <- predict(bst, dtest)
labels <- getinfo(dtest,'label')
cat('error=', mean(as.numeric(preds>0.5)!=labels),'\n')
xgb.save(bst, 'xgb.model')
xgb.dump(bst, 'dump.raw.txt')
xgb.dump(bst, 'dump.nuce.txt','../data/featmap.txt')

bst2 <- xgb.load('xgb.model')
preds2 <- predict(bst2,dtest)
stopifnot(sum((preds-preds2)^2)==0)


cat('start running example of build DMatrix from scipy.sparse CSR Matrix\n')
read.libsvm <- function(fname, maxcol) {
    content <- readLines(fname)
    nline <- length(content)
    label <- numeric(nline)
    mat <- matrix(0, nline, maxcol + 1)
    for (i in 1:nline) {
        arr <- as.vector(strsplit(content[i], " ")[[1]])
        label[i] <- as.numeric(arr[[1]])
        for (j in 2:length(arr)) {
            kv <- strsplit(arr[j], ":")[[1]]
            # to avoid 0 index
            findex <- as.integer(kv[1]) + 1
            fvalue <- as.numeric(kv[2])
            mat[i, findex] <- fvalue
        }
    }
    mat <- as(mat, "sparseMatrix")
    return(list(label = label, data = mat))
}
csc <- read.libsvm("../data/agaricus.txt.train", 126)
y <- csc$label
x <- csc$data
class(x)
dtrain <- xgb.DMatrix(x, label = y)
bst <- xgb.train(param, dtrain, num_round, watchlist)

cat('start running example of build DMatrix from numpy array\n')
x <- as.matrix(x)
class(x)
dtrain <- xgb.DMatrix(x, label = y)
bst <- xgb.train(param, dtrain, num_round, watchlist)

