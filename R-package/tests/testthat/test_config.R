context('Test global configuration')

require(xgboost)

test_that('Global configuration works with verbosity', {
  old_verbosity <- xgb.get.config()$verbosity
  for (v in c(0, 1, 2, 3)) {
    xgb.set.config(verbosity = v)
    expect_equal(xgb.get.config()$verbosity, v)
  }
  xgb.set.config(verbosity = old_verbosity)
  expect_equal(xgb.get.config()$verbosity, old_verbosity)
})
