require(xgboost)

# helper function to read libsvm format
# this is very badly written, load in dense, and convert to sparse
# use this only for demo purpose
# adopted from https://github.com/zygmuntz/r-libsvm-format-read-write/blob/master/f_read.libsvm.r
read.libsvm = function(fname, maxcol) {
    content = readLines(fname)
    nline = length(content)
    label = numeric(nline)
    mat = matrix(0, nline, maxcol+1)
    for (i in 1:nline) {
        arr = as.vector(strsplit(content[i], " ")[[1]])
        label[i] = as.numeric(arr[[1]])
        for (j in 2:length(arr)) {
            kv = strsplit(arr[j], ":")[[1]]
            # to avoid 0 index
            findex = as.integer(kv[1]) + 1
            fvalue = as.numeric(kv[2])
            mat[i,findex] = fvalue
        }
    }
    mat = as(mat, "sparseMatrix")
    return(list(label=label, data=mat))
}

# Parameter setting
dtrain <- xgb.DMatrix("agaricus.txt.train")
dtest <- xgb.DMatrix("agaricus.txt.test")
param = list("bst:max_depth"=2, "bst:eta"=1, "silent"=1, "objective"="binary:logistic")
watchlist = list("eval"=dtest,"train"=dtrain)

###########################
# Train from local file
###########################

# Training
bst = xgboost(file='agaricus.txt.train',params=param,watchlist=watchlist)
# Prediction
pred = predict(bst, 'agaricus.txt.test')
# Performance
labels = xgb.getinfo(dtest, "label")
err = as.numeric(sum(as.integer(pred > 0.5) != labels)) / length(labels)
print(paste("error=",err))

###########################
# Train from R object
###########################

csc = read.libsvm("agaricus.txt.train", 126)
y = csc$label
x = csc$data
# x as Sparse Matrix
class(x)

# Training
bst = xgboost(x,y,params=param,watchlist=watchlist)
# Prediction
pred = predict(bst, 'agaricus.txt.test')
# Performance
labels = xgb.getinfo(dtest, "label")
err = as.numeric(sum(as.integer(pred > 0.5) != labels)) / length(labels)
print(paste("error=",err))

# Training with dense matrix
x = as.matrix(x)
bst = xgboost(x,y,params=param,watchlist=watchlist)

###########################
# Train with customization
###########################

# user define objective function, given prediction, return gradient and second order gradient
# this is loglikelihood loss
logregobj = function(preds, dtrain) {
    labels = xgb.getinfo(dtrain, "label")
    preds = 1.0 / (1.0 + exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return(list(grad=grad, hess=hess))
}
# user defined evaluation function, return a list(metric="metric-name", value="metric-value")
# NOTE: when you do customized loss function, the default prediction value is margin
# this may make buildin evalution metric not function properly
# for example, we are doing logistic loss, the prediction is score before logistic transformation
# the buildin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need write customized evaluation function
evalerror = function(preds, dtrain) {
    labels = xgb.getinfo(dtrain, "label")
    err = as.numeric(sum(labels != (preds > 0.0))) / length(labels)
    return(list(metric="error", value=err))
}

bst = xgboost(x,y,params=param,watchlist=watchlist,obj=logregobj, feval=evalerror)

############################
# Train with previous result
############################

bst = xgboost(x,y,params=param,watchlist=watchlist)
pred = predict(bst, 'agaricus.txt.train', outputmargin=TRUE)
bst2 = xgboost(x,y,params=param,watchlist=watchlist,margin=pred)
