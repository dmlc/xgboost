require(xgboost)

context("interaction constraints")

set.seed(1024)
x1 <- rnorm(1000, 1)
x2 <- rnorm(1000, 1)
x3 <- sample(c(1,2,3), size=1000, replace=TRUE)
y <- x1 + x2 + x3 + x1*x2*x3 + rnorm(1000, 0.001) + 3*sin(x1)
train <- matrix(c(x1,x2,x3), ncol = 3)

test_that("interaction constraints for regression", {
  # Fit a model that only allows interaction between x1 and x2
  bst <- xgboost(data = train, label = y, max_depth = 3,
                 eta = 0.1, nthread = 2, nrounds = 100, verbose = 0,
                 interaction_constraints = list(c(0,1)))
  
  # Set all observations to have the same x3 values then increment
  #  by the same amount
	preds <- lapply(c(1,2,3), function(x){
		tmat <- matrix(c(x1,x2,rep(x,1000)), ncol=3)
		return(predict(bst, tmat))
	})

  # Check incrementing x3 has the same effect on all observations
  #   since x3 is constrained to be independent of x1 and x2
  #   and all observations start off from the same x3 value
	diff1 <- preds[[2]] - preds[[1]]
	test1 <- all(abs(diff1 - diff1[1]) < 1e-4)
	
	diff2 <- preds[[3]] - preds[[2]]
	test2 <- all(abs(diff2 - diff2[1]) < 1e-4)
	
  expect_true({
    test1 & test2
  }, "Interaction Contraint Satisfied")
  
})
