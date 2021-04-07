context('Test global configuration')

test_that('Global configuration works with verbosity', {
  old_verbosity <- xgb.get.config()$verbosity
  for (v in c(0, 1, 2, 3)) {
    xgb.set.config(verbosity = v)
    expect_equal(xgb.get.config()$verbosity, v)
  }
  xgb.set.config(verbosity = old_verbosity)
  expect_equal(xgb.get.config()$verbosity, old_verbosity)
})

test_that('Global configuration works with use_rmm flag', {
  old_use_rmm_flag <- xgb.get.config()$use_rmm
  for (v in c(TRUE, FALSE)) {
    xgb.set.config(use_rmm = v)
    expect_equal(xgb.get.config()$use_rmm, v)
  }
  xgb.set.config(use_rmm = old_use_rmm_flag)
  expect_equal(xgb.get.config()$use_rmm, old_use_rmm_flag)
})
