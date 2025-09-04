context("Models from previous versions of XGBoost can be loaded")

metadata <- list(
  kRounds = 4,
  kRows = 1000,
  kCols = 4,
  kForests = 2,
  kMaxDepth = 2,
  kClasses = 3
)

run_model_param_check <- function(config) {
  testthat::expect_equal(config$learner$learner_model_param$num_feature, "4")
  testthat::expect_equal(config$learner$learner_train_param$booster, "gbtree")
}

get_n_rounds <- function(model_file) {
  is_10 <- grepl("1.0.0rc1", model_file, fixed = TRUE)
  if (is_10) {
    2
  } else {
    metadata$kRounds
  }
}

get_num_tree <- function(booster) {
  dump <- xgb.dump(booster)
  m <- regexec("booster\\[[0-9]+\\]", dump, perl = TRUE)
  m <- regmatches(dump, m)
  num_tree <- Reduce("+", lapply(m, length))
  num_tree
}

run_booster_check <- function(booster, model_file) {
  config <- xgb.config(booster)
  run_model_param_check(config)
  is_model <- function(typ) {
    grepl(typ, model_file, fixed = TRUE)
  }
  n_rounds <- get_n_rounds(model_file = model_file)
  if (is_model("cls")) {
    testthat::expect_equal(
      get_num_tree(booster), metadata$kForests * n_rounds * metadata$kClasses
    )
    testthat::expect_equal(get_basescore(config), c(0.5, 0.5, 0.5))  # nolint
    testthat::expect_equal(
      config$learner$learner_train_param$objective, "multi:softmax"
    )
    testthat::expect_equal(
      as.numeric(config$learner$learner_model_param$num_class),
      metadata$kClasses
    )
  } else if (is_model("logitraw")) {
    testthat::expect_equal(get_num_tree(booster), metadata$kForests * n_rounds)
    testthat::expect_equal(
      as.numeric(config$learner$learner_model_param$num_class), 0
    )
    testthat::expect_equal(
      config$learner$learner_train_param$objective, "binary:logitraw"
    )
  } else if (is_model("logit")) {
    testthat::expect_equal(get_num_tree(booster), metadata$kForests * n_rounds)
    testthat::expect_equal(
      as.numeric(config$learner$learner_model_param$num_class), 0
    )
    testthat::expect_equal(
      config$learner$learner_train_param$objective, "binary:logistic"
    )
  } else if (is_model("ltr")) {
    testthat::expect_equal(get_num_tree(booster), metadata$kForests * n_rounds)
    testthat::expect_equal(
      config$learner$learner_train_param$objective, "rank:ndcg"
    )
  } else if (is_model("aft")) {
    testthat::expect_equal(get_num_tree(booster), metadata$kForests * n_rounds)
    testthat::expect_equal(
      config$learner$learner_train_param$objective, "survival:aft"
    )
  } else {
    testthat::expect_true(is_model("reg"))
    testthat::expect_equal(get_num_tree(booster), metadata$kForests * n_rounds)
    testthat::expect_equal(get_basescore(config), 0.5)  # nolint
    testthat::expect_equal(
      config$learner$learner_train_param$objective, "reg:squarederror"
    )
  }
}

test_that("Models from previous versions of XGBoost can be loaded", {
  bucket <- "xgboost-ci-jenkins-artifacts"
  region <- "us-west-2"
  file_name <- "xgboost_model_compatibility_tests-3.0.2.zip"
  zipfile <- tempfile(fileext = ".zip")
  extract_dir <- tempdir()
  result <- tryCatch(
    {
      download.file(
        paste(
          "https://", bucket, ".s3-", region, ".amazonaws.com/", file_name,
          sep = ""
        ),
        destfile = zipfile, mode = "wb", quiet = TRUE
      )
      zipfile
    },
    error = function(e) {
      print(e)
      NA_character_
    }
  )
  if (is.na(result)) {
    print("Failed to download old models.")
    return()
  }

  unzip(zipfile, exdir = extract_dir, overwrite = TRUE)
  model_dir <- file.path(extract_dir, "models")

  pred_data <- xgb.DMatrix(
    matrix(c(0, 0, 0, 0), nrow = 1, ncol = 4),
    nthread = 2
  )

  lapply(list.files(model_dir), function(x) {
    model_file <- file.path(model_dir, x)
    is_skl <- grepl("scikit", model_file, fixed = TRUE)
    if (is_skl) {
      return()
    }
    booster <- xgb.load(model_file)
    xgb.model.parameters(booster) <- list(nthread = 2)
    predict(booster, newdata = pred_data)
    run_booster_check(booster, model_file)
  })
})
