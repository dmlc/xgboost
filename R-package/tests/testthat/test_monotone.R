context("monotone constraints")

set.seed(1024)
x <- rnorm(1000, 10)
y <- -1 * x + rnorm(1000, 0.001) + 3 * sin(x)
train <- matrix(x, ncol = 1)


test_that("monotone constraints for regression", {
    bst <- xgboost(data = train, label = y, max_depth = 2,
                   eta = 0.1, nthread = 2, nrounds = 100, verbose = 0,
                   monotone_constraints = -1)

    pred <- predict(bst, train)

    ind <- order(train[, 1])
    pred.ord <- pred[ind]
    expect_true({
        !any(diff(pred.ord) > 0)
    }, "Monotone constraint satisfied")
})
