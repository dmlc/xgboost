context("monotone constraints")

set.seed(1024)
x <- rnorm(1000, 10)
y <- -1 * x + rnorm(1000, 0.001) + 3 * sin(x)
train <- matrix(x, ncol = 1)


test_that("monotone constraints for regression", {
    bst <- xgb.train(
        data = xgb.DMatrix(train, label = y),
        nrounds = 100, verbose = 0,
        params = xgb.params(
            max_depth = 2,
            learning_rate = 0.1,
            nthread = 2,
            monotone_constraints = -1
        )
    )

    pred <- predict(bst, train)

    ind <- order(train[, 1])
    pred.ord <- pred[ind]
    expect_true({
        !any(diff(pred.ord) > 0)
    }, "Monotone constraint satisfied")
})
