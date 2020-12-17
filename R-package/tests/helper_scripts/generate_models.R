# Script to generate reference models. The reference models are used to test backward compatibility
# of saved model files from XGBoost version 0.90 and 1.0.x.
library(xgboost)
library(Matrix)

set.seed(0)
metadata <- list(
  kRounds = 2,
  kRows = 1000,
  kCols = 4,
  kForests = 2,
  kMaxDepth = 2,
  kClasses = 3
)
X <- Matrix(data = rnorm(metadata$kRows * metadata$kCols), nrow = metadata$kRows,
            ncol = metadata$kCols, sparse = TRUE)
w <- runif(metadata$kRows)

version <- packageVersion('xgboost')
target_dir <- 'models'

save_booster <- function (booster, model_name) {
  booster_bin <- function (model_name) {
    return (file.path(target_dir, paste('xgboost-', version, '.', model_name, '.bin', sep = '')))
  }
  booster_json <- function (model_name) {
    return (file.path(target_dir, paste('xgboost-', version, '.', model_name, '.json', sep = '')))
  }
  booster_rds <- function (model_name) {
    return (file.path(target_dir, paste('xgboost-', version, '.', model_name, '.rds', sep = '')))
  }
  xgb.save(booster, booster_bin(model_name))
  saveRDS(booster, booster_rds(model_name))
  if (version >= '1.0.0') {
    xgb.save(booster, booster_json(model_name))
  }
}

generate_regression_model <- function () {
  print('Regression')
  y <- rnorm(metadata$kRows)

  data <- xgb.DMatrix(X, label = y)
  params <- list(tree_method = 'hist', num_parallel_tree = metadata$kForests,
                 max_depth = metadata$kMaxDepth)
  booster <- xgb.train(params, data, nrounds = metadata$kRounds)
  save_booster(booster, 'reg')
}

generate_logistic_model <- function () {
  print('Binary classification with logistic loss')
  y <- sample(0:1, size = metadata$kRows, replace = TRUE)
  stopifnot(max(y) == 1, min(y) == 0)

  objective <- c('binary:logistic', 'binary:logitraw')
  name <- c('logit', 'logitraw')

  for (i in seq_len(length(objective))) {
    data <- xgb.DMatrix(X, label = y, weight = w)
    params <- list(tree_method = 'hist', num_parallel_tree = metadata$kForests,
                   max_depth = metadata$kMaxDepth, objective = objective[i])
    booster <- xgb.train(params, data, nrounds = metadata$kRounds)
    save_booster(booster, name[i])
  }
}

generate_classification_model <- function () {
  print('Multi-class classification')
  y <- sample(0:(metadata$kClasses - 1), size = metadata$kRows, replace = TRUE)
  stopifnot(max(y) == metadata$kClasses - 1, min(y) == 0)

  data <- xgb.DMatrix(X, label = y, weight = w)
  params <- list(num_class = metadata$kClasses, tree_method = 'hist',
                 num_parallel_tree = metadata$kForests, max_depth = metadata$kMaxDepth,
                 objective = 'multi:softmax')
  booster <- xgb.train(params, data, nrounds = metadata$kRounds)
  save_booster(booster, 'cls')
}

generate_ranking_model <- function () {
  print('Learning to rank')
  y <- sample(0:4, size = metadata$kRows, replace = TRUE)
  stopifnot(max(y) == 4, min(y) == 0)
  kGroups <- 20
  w <- runif(kGroups)
  g <- rep(50, times = kGroups)

  data <- xgb.DMatrix(X, label = y, group = g)
  # setinfo(data, 'weight', w)
  # ^^^ does not work in version <= 1.1.0; see https://github.com/dmlc/xgboost/issues/5942
  # So call low-level function XGDMatrixSetInfo_R directly. Since this function is not an exported
  # symbol, use the triple-colon operator.
  .Call(xgboost:::XGDMatrixSetInfo_R, data, 'weight', as.numeric(w))
  params <- list(objective = 'rank:ndcg', num_parallel_tree = metadata$kForests,
                 tree_method = 'hist', max_depth = metadata$kMaxDepth)
  booster <- xgb.train(params, data, nrounds = metadata$kRounds)
  save_booster(booster, 'ltr')
}

dir.create(target_dir)

invisible(generate_regression_model())
invisible(generate_logistic_model())
invisible(generate_classification_model())
invisible(generate_ranking_model())
