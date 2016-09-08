require(xgboost)

context("monotone constraints")

set.seed(1024)
x1 = rnorm(1000, 10)
x2 = rnorm(1000, 10)
y = -1*x1 + rnorm(1000, 0.001) + 3*sin(x2)
train = cbind(x1, x2)


test_that("monotone constraints for regression", {
  nrounds = 10
  expect_output(
    bst <- xgboost(data = train, label = y, max_depth = 2,
                   eta = 0.1, nthread = 2, nrounds = nrounds,
                   monotone_constraints = c(1,-1))
    , "monotone-error")
  
  pred <- predict(bst, train)
  
  ind = order(train[,1])
  pred.ord = pred[ind]
  expect_true({
    !any(diff(pred.ord) < 0)
  }, "Monotone Contraint Satisfied")
  
  ind = order(train[,2])
  pred.ord = pred[ind]
  expect_true({
    !any(diff(pred.ord) > 0)
  }, "Monotone Contraint Satisfied")
})
