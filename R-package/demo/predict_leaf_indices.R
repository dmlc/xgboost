require(xgboost)
require(data.table)
require(Matrix)

set.seed(1982)

# load in the agaricus dataset
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
dtrain <- xgb.DMatrix(data = agaricus.train$data, label = agaricus.train$label)
dtest <- xgb.DMatrix(data = agaricus.test$data, label = agaricus.test$label)

param <- list(max_depth=2, eta=1, objective='binary:logistic')
nrounds = 4

# training the model for two rounds
bst = xgb.train(params = param, data = dtrain, nrounds = nrounds, nthread = 2)

# Model accuracy without new features
accuracy.before <- sum((predict(bst, agaricus.test$data) >= 0.5) == agaricus.test$label) / length(agaricus.test$label)

# by default, we predict using all the trees

pred_with_leaf = predict(bst, dtest, predleaf = TRUE)
head(pred_with_leaf)

create.new.tree.features <- function(model, original.features){
  pred_with_leaf <- predict(model, original.features, predleaf = TRUE)
  cols <- list()
  for(i in 1:model$niter){
    # max is not the real max but it s not important for the purpose of adding features
    leaf.id <- sort(unique(pred_with_leaf[,i]))
    cols[[i]] <- factor(x = pred_with_leaf[,i], level = leaf.id)
  }
  cbind(original.features, sparse.model.matrix( ~ . -1, as.data.frame(cols)))
}

# Convert previous features to one hot encoding
new.features.train <- create.new.tree.features(bst, agaricus.train$data)
new.features.test <- create.new.tree.features(bst, agaricus.test$data)
colnames(new.features.test) <- colnames(new.features.train)

# learning with new features
new.dtrain <- xgb.DMatrix(data = new.features.train, label = agaricus.train$label)
new.dtest <- xgb.DMatrix(data = new.features.test, label = agaricus.test$label)
watchlist <- list(train = new.dtrain)
bst <- xgb.train(params = param, data = new.dtrain, nrounds = nrounds, nthread = 2)

# Model accuracy with new features
accuracy.after <- sum((predict(bst, new.dtest) >= 0.5) == agaricus.test$label) / length(agaricus.test$label)

# Here the accuracy was already good and is now perfect.
cat(paste("The accuracy was", accuracy.before, "before adding leaf features and it is now", accuracy.after, "!\n"))
