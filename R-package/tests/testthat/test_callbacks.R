# More specific testing of callbacks

require(xgboost)
require(data.table)

context("callbacks")

data(agaricus.train, package='xgboost')
train <- agaricus.train

dtrain <- xgb.DMatrix(agaricus.train$data, label = agaricus.train$label)
watchlist = list(train=dtrain)
param <- list(objective = "binary:logistic", max.depth = 2, nthread = 2)


test_that("cb.print_evaluation works as expected", {
  
  bst_evaluation <- c('train-auc'=0.9, 'test-auc'=0.8)
  bst_evaluation_err <- NULL
  begin_iteration <- 1
  end_iteration <- 7
  
  f0 <- cb.print_evaluation(period=0)
  f1 <- cb.print_evaluation(period=1)
  f5 <- cb.print_evaluation(period=5)
  
  expect_true(!is.null(attr(f1, 'call')))
  expect_equal(attr(f1, 'name'), 'cb.print_evaluation')

  iteration <- 1
  expect_silent(f0())
  expect_output(f1(), "\\[1\\]\ttrain-auc:0.900000\ttest-auc:0.800000")
  expect_output(f5(), "\\[1\\]\ttrain-auc:0.900000\ttest-auc:0.800000")
  expect_null(f1())
  
  iteration <- 2
  expect_output(f1(), "\\[2\\]\ttrain-auc:0.900000\ttest-auc:0.800000")
  expect_silent(f5())
  
  iteration <- 7
  expect_output(f1(), "\\[7\\]\ttrain-auc:0.900000\ttest-auc:0.800000")
  expect_output(f5(), "\\[7\\]\ttrain-auc:0.900000\ttest-auc:0.800000")
  
  bst_evaluation_err  <- c('train-auc'=0.1, 'test-auc'=0.2)
  expect_output(f1(), "\\[7\\]\ttrain-auc:0.900000\\+0.100000\ttest-auc:0.800000\\+0.200000")
})

test_that("cb.log_evaluation works as expected", {

  bst_evaluation <- c('train-auc'=0.9, 'test-auc'=0.8)
  bst_evaluation_err <- NULL
  
  evaluation_log <- list()
  f <- cb.log_evaluation()
  
  expect_true(!is.null(attr(f, 'call')))
  expect_equal(attr(f, 'name'), 'cb.log_evaluation')
  
  iteration <- 1
  expect_silent(f())
  expect_equal(evaluation_log, 
               list(c(iter=1, bst_evaluation)))
  iteration <- 2
  expect_silent(f())
  expect_equal(evaluation_log, 
               list(c(iter=1, bst_evaluation), c(iter=2, bst_evaluation)))
  expect_silent(f(finalize = TRUE))
  expect_equal(evaluation_log, 
               data.table(iter=1:2, train_auc=c(0.9,0.9), test_auc=c(0.8,0.8)))
  
  bst_evaluation_err  <- c('train-auc'=0.1, 'test-auc'=0.2)
  evaluation_log <- list()
  f <- cb.log_evaluation()
  
  iteration <- 1
  expect_silent(f())
  expect_equal(evaluation_log, 
               list(c(iter=1, c(bst_evaluation, bst_evaluation_err))))
  iteration <- 2
  expect_silent(f())
  expect_equal(evaluation_log, 
               list(c(iter=1, c(bst_evaluation, bst_evaluation_err)),
                    c(iter=2, c(bst_evaluation, bst_evaluation_err))))
  expect_silent(f(finalize = TRUE))
  expect_equal(evaluation_log, 
               data.table(iter=1:2,
                          train_auc_mean=c(0.9,0.9), train_auc_std=c(0.1,0.1),
                          test_auc_mean=c(0.8,0.8), test_auc_std=c(0.2,0.2)))
})

test_that("cb.reset_parameters works as expected", {

  # fixed eta
  set.seed(111)
  bst0 <- xgb.train(param, dtrain, nrounds = 2, watchlist, eta = 0.9)
  expect_true(!is.null(bst0$evaluation_log))
  expect_true(!is.null(bst0$evaluation_log$train_error))

  # same eta but re-set as a vector parameter in the callback
  set.seed(111)
  my_par <- list(eta = c(0.9, 0.9))
  bst1 <- xgb.train(param, dtrain, nrounds = 2, watchlist,
                    callbacks = list(cb.reset_parameters(my_par)))
  expect_true(!is.null(bst1$evaluation_log$train_error))
  expect_equal(bst0$evaluation_log$train_error, 
               bst1$evaluation_log$train_error)
  
  # same eta but re-set via a function in the callback
  set.seed(111)
  my_par <- list(eta = function(itr, itr_end) 0.9)
  bst2 <- xgb.train(param, dtrain, nrounds = 2, watchlist,
                    callbacks = list(cb.reset_parameters(my_par)))
  expect_true(!is.null(bst2$evaluation_log$train_error))
  expect_equal(bst0$evaluation_log$train_error, 
               bst2$evaluation_log$train_error)
  
  # different eta re-set as a vector parameter in the callback
  set.seed(111)
  my_par <- list(eta = c(0.6, 0.5))
  bst3 <- xgb.train(param, dtrain, nrounds = 2, watchlist,
                    callbacks = list(cb.reset_parameters(my_par)))
  expect_true(!is.null(bst3$evaluation_log$train_error))
  expect_true(!all(bst0$evaluation_log$train_error == bst3$evaluation_log$train_error))
  
  # resetting multiple parameters at the same time runs with no error
  my_par <- list(eta = c(1., 0.5), gamma = c(1, 2), alpha = c(0.01, 0.02))
  expect_error(
    bst4 <- xgb.train(param, dtrain, nrounds = 2, watchlist,
                      callbacks = list(cb.reset_parameters(my_par)))
  , NA)

  # expect no learning with 0 learning rate
  my_par <- list(eta = c(0., 0.))
  bstX <- xgb.train(param, dtrain, nrounds = 2, watchlist, 
                    callbacks = list(cb.reset_parameters(my_par)))
  expect_true(!is.null(bstX$evaluation_log$train_error))
  er <- unique(bstX$evaluation_log$train_error)
  expect_length(er, 1)
  expect_gt(er, 0.4)
})

# Note: early stopping is tested in test_basic

test_that("cb.save_model works as expected", {
  files <- c('xgboost_01.model', 'xgboost_02.model', 'xgboost.model')
  for (f in files) if (file.exists(f)) file.remove(f)
  
  bst <- xgb.train(param, dtrain, nrounds = 2, watchlist,
                   save_period = 1, save_name = "xgboost_%02d.model")
  expect_true(file.exists('xgboost_01.model'))
  expect_true(file.exists('xgboost_02.model'))
  b1 <- xgb.load('xgboost_01.model')
  expect_length(grep('^booster', xgb.dump(b1)), 1)
  b2 <- xgb.load('xgboost_02.model')
  expect_equal(bst$raw, b2$raw)

  # save_period = 0 saves the last iteration's model
  bst <- xgb.train(param, dtrain, nrounds = 2, watchlist, save_period = 0)
  expect_true(file.exists('xgboost.model'))
  b2 <- xgb.load('xgboost.model')
  expect_equal(bst$raw, b2$raw)
  
  for (f in files) if (file.exists(f)) file.remove(f)
})

test_that("can store evaluation_log without printing", {
  expect_silent(
    bst <- xgb.train(param, dtrain, nrounds = 2, watchlist, eta = 1,
                     verbose = 0, callbacks = list(cb.log_evaluation()))
  )
  expect_true(!is.null(bst$evaluation_log))
  expect_true(!is.null(bst$evaluation_log$train_error))
  expect_lt(bst$evaluation_log[2, train_error], 0.03)
})
