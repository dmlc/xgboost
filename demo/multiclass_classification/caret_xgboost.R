data("iris")
require(caret)
require(xgboost)
require(doMC)
registerDoMC(cores = 4)
set.seed(2016)
n <- dim(iris)[1]
train_id <- sort(sample(1:n, ceiling(n * 0.7), FALSE))
train_X <- iris[train_id, 1:4]
train_Y <- iris[train_id, 5]

test_X <- iris[-train_id, 1:4]
test_Y <- iris[-train_id, 5]

xgb_grid <- expand.grid(nrounds = c(2000, 1000, 500),
                        eta = c(0.01, 0.1, 0.1),
                        max_depth = c(2, 4, 8, 16, 32),
                        gamma = 0.1,
                        colsample_bytree = 1,
                        min_child_weight = 1)

xgb_trCtrol <- trainControl(method = 'cv',
                            number = 5,
                            classProbs = TRUE,
                            summaryFunction = multiClassSummary,
                            allowParallel = F)

xgb_model <- train(x = train_X,
                   y = train_Y,
                   trControl = xgb_trCtrol,
                   tuneGrid = xgb_grid,
                   method = 'xgbTree',
                   preProcess = c('center', 'scale'),
                   metric = 'logLoss')

plot(xgb_model)

pred <- predict(xgb_model, test_X, type = 'prob')
pred$obs <- test_Y
print(mnLogLoss(pred, lev = levels(pred$obs)))
