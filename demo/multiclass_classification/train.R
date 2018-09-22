library(data.table)
library(xgboost)

if (!file.exists("./dermatology.data")) {
  download.file(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data",
    "dermatology.data",
    method = "curl"
  )
}

df <- fread("dermatology.data", sep = ",", header = FALSE)

df[, `:=`(V34 = as.integer(ifelse(V34 == "?", 0L, V34)),
          V35 = V35 - 1L)]

idx <- sample(nrow(df), size = round(0.7 * nrow(df)), replace = FALSE)

train <- df[idx,]
test <- df[-idx,]

train_x <- train[, 1:34]
train_y <- train[, V35]

test_x <- test[, 1:34]
test_y <- test[, V35]

xg_train <- xgb.DMatrix(data = as.matrix(train_x), label = train_y)
xg_test = xgb.DMatrix(as.matrix(test_x), label = test_y)

params <- list(
  objective = 'multi:softmax',
  num_class = 6,
  max_depth = 6,
  nthread = 4,
  eta = 0.1
)

watchlist = list(train = xg_train, test = xg_test)

bst <- xgb.train(
  params = params,
  data = xg_train,
  watchlist = watchlist,
  nrounds = 5
)

pred <- predict(bst, xg_test)
error_rate <- sum(pred != test_y) / length(test_y)
print(paste("Test error using softmax =", error_rate))

# do the same thing again, but output probabilities
params$objective <- 'multi:softprob'
bst <- xgb.train(params, xg_train, nrounds = 5, watchlist)

pred_prob <- predict(bst, xg_test)

pred_mat <- matrix(pred_prob, ncol = 6, byrow = TRUE)
# validation
# rowSums(pred_mat)

pred_label <- apply(pred_mat, 1, which.max) - 1L
error_rate = sum(pred_label != test_y) / length(test_y)
print(paste("Test error using softprob =", error_rate))
