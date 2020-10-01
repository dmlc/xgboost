library(xgboost)
library(data.table)
library(cplm)

data(AutoClaim)

# auto insurance dataset analyzed by Yip and Yau (2005)
dt <- data.table(AutoClaim)

# exclude these columns from the model matrix
exclude <- c('POLICYNO', 'PLCYDATE', 'CLM_FREQ5', 'CLM_AMT5', 'CLM_FLAG', 'IN_YY')

# retains the missing values
# NOTE: this dataset is comes ready out of the box
options(na.action = 'na.pass')
x <- sparse.model.matrix(~ . - 1, data = dt[, -exclude, with = FALSE])
options(na.action = 'na.omit')

# response
y <- dt[, CLM_AMT5]

d_train <- xgb.DMatrix(data = x, label = y, missing = NA)

# the tweedie_variance_power parameter determines the shape of
# distribution
# - closer to 1 is more poisson like and the mass
#   is more concentrated near zero
# - closer to 2 is more gamma like and the mass spreads to the
#   the right with less concentration near zero

params <- list(
  objective = 'reg:tweedie',
  eval_metric = 'rmse',
  tweedie_variance_power = 1.4,
  max_depth = 6,
  eta = 1)

bst <- xgb.train(
  data = d_train,
  params = params,
  maximize = FALSE,
  watchlist = list(train = d_train),
  nrounds = 20)

var_imp <- xgb.importance(attr(x, 'Dimnames')[[2]], model = bst)

preds <- predict(bst, d_train)

rmse <- sqrt(sum(mean((y - preds) ^ 2)))
