# An example of using GPU-accelerated tree building algorithms
#
# NOTE: it can only run if you have a CUDA-enable GPU and the package was
#       specially compiled with GPU support.
#
# For the current functionality, see
# https://xgboost.readthedocs.io/en/latest/gpu/index.html
#

library("xgboost")

# Simulate N x p random matrix with some binomial response dependent on pp columns
set.seed(111)
N <- 1000000
p <- 50
pp <- 25
X <- matrix(runif(N * p), ncol = p)
betas <- 2 * runif(pp) - 1
sel <- sort(sample(p, pp))
m <- X[, sel] %*% betas - 1 + rnorm(N)
y <- rbinom(N, 1, plogis(m))

tr <- sample.int(N, N * 0.75)
dtrain <- xgb.DMatrix(X[tr, ], label = y[tr])
dtest <- xgb.DMatrix(X[-tr, ], label = y[-tr])
evals <- list(train = dtrain, test = dtest)

# An example of running gpu-accelerated 'hist' algorithm
# which is
# - similar to the 'hist'
# - the fastest option for moderately large datasets
# - current limitations: max_depth < 16, does not implement guided loss


pt <- proc.time()
bst_gpu <- xgb.train(
    data = dtrain,
    watchlist = list(train = dtrain, test = dtest),
    objective = "reg:logistic",
    eval_metric = "auc",
    subsample = 0.5,
    nthread = 4,
    nround = 50,
    print_every_n = 50,
    max_bin = 64,
    tree_method = "hist", # or "gpu_hist" for xgboost < 2.0.0
    device = "cuda" # Since xgboost 2.0.0 the device argument is introduced
)
proc.time() - pt

# Compare to the 'hist' algorithm:

pt <- proc.time()
bst_hist <- xgb.train(
    data = dtrain,
    watchlist = list(train = dtrain, test = dtest),
    objective = "reg:logistic",
    eval_metric = "auc",
    subsample = 0.5,
    nthread = 4,
    nround = 50,
    print_every_n = 50,
    tree_method = "hist",
    max_bin = 64
)
proc.time() - pt
