# An example of using GPU-accelerated tree building algorithms
#
# NOTE: it can only run if you have a CUDA-enable GPU and the package was
#       specially compiled with GPU support.
#
# For the current functionality, see
# https://xgboost.readthedocs.io/en/latest/gpu/index.html
#

library('xgboost')

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
wl <- list(train = dtrain, test = dtest)

# An example of running 'gpu_hist' algorithm
# which is
# - similar to the 'hist'
# - the fastest option for moderately large datasets
# - current limitations: max_depth < 16, does not implement guided loss
# You can use tree_method = 'gpu_hist' for another GPU accelerated algorithm,
# which is slower, more memory-hungry, but does not use binning.
param <- list(objective = 'reg:logistic', eval_metric = 'auc', subsample = 0.5, nthread = 4,
              max_bin = 64, tree_method = 'gpu_hist')
pt <- proc.time()
bst_gpu <- xgb.train(param, dtrain, watchlist = wl, nrounds = 50)
proc.time() - pt

# Compare to the 'hist' algorithm:
param$tree_method <- 'hist'
pt <- proc.time()
bst_hist <- xgb.train(param, dtrain, watchlist = wl, nrounds = 50)
proc.time() - pt
