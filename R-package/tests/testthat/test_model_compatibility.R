require(xgboost)
require(jsonlite)
source('../generate_models_params.R')

context("Models from previous versions of XGBoost can be loaded")

metadata <- model_generator_metadata()

run_model_param_check <- function (config) {
  expect_equal(config$learner$learner_model_param$num_feature, '4')
  expect_equal(config$learner$learner_train_param$booster, 'gbtree')
}

get_num_tree <- function (booster) {
  dump <- xgb.dump(booster)
  m <- regexec('booster\\[[0-9]+\\]', dump, perl = TRUE)
  m <- regmatches(dump, m)
  num_tree <- Reduce('+', lapply(m, length))
  return (num_tree)
}

run_booster_check <- function (booster, name) {
  # If given a handle, we need to call xgb.Booster.complete() prior to using xgb.config().
  if (inherits(booster, "xgb.Booster") && xgboost:::is.null.handle(booster$handle)) {
    booster <- xgb.Booster.complete(booster)
  }
  config <- jsonlite::fromJSON(xgb.config(booster))
  run_model_param_check(config)
  if (name == 'cls') {
    expect_equal(get_num_tree(booster), metadata$kForests * metadata$kRounds * metadata$kClasses)
    expect_equal(as.numeric(config$learner$learner_model_param$base_score), 0.5)
    expect_equal(config$learner$learner_train_param$objective, 'multi:softmax')
    expect_equal(as.numeric(config$learner$learner_model_param$num_class), metadata$kClasses)
  } else if (name == 'logit') {
    expect_equal(get_num_tree(booster), metadata$kForests * metadata$kRounds)
    expect_equal(as.numeric(config$learner$learner_model_param$num_class), 0)
    expect_equal(config$learner$learner_train_param$objective, 'binary:logistic')
  } else if (name == 'ltr') {
    expect_equal(get_num_tree(booster), metadata$kForests * metadata$kRounds)
    expect_equal(config$learner$learner_train_param$objective, 'rank:ndcg')
  } else {
    expect_equal(name, 'reg')
    expect_equal(get_num_tree(booster), metadata$kForests * metadata$kRounds)
    expect_equal(as.numeric(config$learner$learner_model_param$base_score), 0.5)
    expect_equal(config$learner$learner_train_param$objective, 'reg:squarederror')
  }
}

test_that("Models from previous versions of XGBoost can be loaded", {
  bucket <- 'xgboost-ci-jenkins-artifacts'
  region <- 'us-west-2'
  file_name <- 'xgboost_r_model_compatibility_test.zip'
  zipfile <- file.path(getwd(), file_name)
  model_dir <- file.path(getwd(), 'models')
  download.file(paste('https://', bucket, '.s3-', region, '.amazonaws.com/', file_name, sep = ''),
                destfile = zipfile, mode = 'wb')
  unzip(zipfile, overwrite = TRUE)

  pred_data <- xgb.DMatrix(matrix(c(0, 0, 0, 0), nrow = 1, ncol = 4))

  lapply(list.files(model_dir), function (x) {
    model_file <- file.path(model_dir, x)
    m <- regexec("xgboost-([0-9\\.]+)\\.([a-z]+)\\.[a-z]+", model_file, perl = TRUE)
    m <- regmatches(model_file, m)[[1]]
    model_xgb_ver <- m[2]
    name <- m[3]

    if (endsWith(model_file, '.rds')) {
      booster <- readRDS(model_file)
    } else {
      booster <- xgb.load(model_file)
    }
    predict(booster, newdata = pred_data)
    run_booster_check(booster, name)
  })
  expect_true(TRUE)
})
