library(survival)
library(data.table)
data("iris")
data("mtcars")
data("ToothGrowth")

test_that("Auto determine objective", {
  y_num <- seq(1, 10)
  res_num <- process.y.margin.and.objective(y_num, NULL, NULL, NULL)
  expect_equal(res_num$params$objective, "reg:squarederror")

  y_bin <- factor(c('a', 'b', 'a', 'b'), c('a', 'b'))
  res_bin <- process.y.margin.and.objective(y_bin, NULL, NULL, NULL)
  expect_equal(res_bin$params$objective, "binary:logistic")

  y_multi <- factor(c('a', 'b', 'a', 'b', 'c'), c('a', 'b', 'c'))
  res_multi <- process.y.margin.and.objective(y_multi, NULL, NULL, NULL)
  expect_equal(res_multi$params$objective, "multi:softprob")

  y_surv <- Surv(1:10, rep(c(0, 1), 5), type = "right")
  res_surv <- process.y.margin.and.objective(y_surv, NULL, NULL, NULL)
  expect_equal(res_surv$params$objective, "survival:aft")

  y_multicol <- matrix(seq(1, 20), nrow = 5)
  res_multicol <- process.y.margin.and.objective(y_multicol, NULL, NULL, NULL)
  expect_equal(res_multicol$params$objective, "reg:squarederror")
})

test_that("Process vectors", {
  y <- seq(1, 10)
  for (y_inp in list(as.integer(y), as.numeric(y))) {
    res <- process.y.margin.and.objective(y_inp, NULL, "reg:pseudohubererror", NULL)
    expect_equal(
      res$dmatrix_args$label,
      y
    )
    expect_equal(
      res$params$objective,
      "reg:pseudohubererror"
    )
  }
})

test_that("Process factors", {
  y_bin <- factor(c('a', 'b', 'a', 'b'), c('a', 'b'))
  expect_error({
    process.y.margin.and.objective(y_bin, NULL, "multi:softprob", NULL)
  })
  for (bin_obj in c("binary:logistic", "binary:hinge")) {
    for (y_inp in list(y_bin, as.ordered(y_bin))) {
      res_bin <- process.y.margin.and.objective(y_inp, NULL, bin_obj, NULL)
      expect_equal(
        res_bin$dmatrix_args$label,
        c(0, 1, 0, 1)
      )
      expect_equal(
        res_bin$metadata$y_levels,
        c('a', 'b')
      )
      expect_equal(
        res_bin$params$objective,
        bin_obj
      )
    }
  }

  y_bin2 <- factor(c(1, 0, 1, 0), c(1, 0))
  res_bin <- process.y.margin.and.objective(y_bin2, NULL, "binary:logistic", NULL)
  expect_equal(
    res_bin$dmatrix_args$label,
    c(0, 1, 0, 1)
  )
  expect_equal(
    res_bin$metadata$y_levels,
    c("1", "0")
  )

  y_bin3 <- c(TRUE, FALSE, TRUE)
  res_bin <- process.y.margin.and.objective(y_bin3, NULL, "binary:logistic", NULL)
  expect_equal(
    res_bin$dmatrix_args$label,
    c(1, 0, 1)
  )
  expect_equal(
    res_bin$metadata$y_levels,
    c("FALSE", "TRUE")
  )

  y_multi <- factor(c('a', 'b', 'c', 'd', 'a', 'b'), c('a', 'b', 'c', 'd'))
  expect_error({
    process.y.margin.and.objective(y_multi, NULL, "binary:logistic", NULL)
  })
  expect_error({
    process.y.margin.and.objective(y_multi, NULL, "binary:logistic", NULL)
  })
  res_multi <- process.y.margin.and.objective(y_multi, NULL, "multi:softprob", NULL)
  expect_equal(
    res_multi$dmatrix_args$label,
    c(0, 1, 2, 3, 0, 1)
  )
  expect_equal(
    res_multi$metadata$y_levels,
    c('a', 'b', 'c', 'd')
  )
  expect_equal(
    res_multi$params$num_class,
    4
  )
  expect_equal(
    res_multi$params$objective,
    "multi:softprob"
  )
})

test_that("Process survival objects", {
  data(cancer, package = "survival")
  y_right <- Surv(cancer$time, cancer$status - 1, type = "right")
  res_cox <- process.y.margin.and.objective(y_right, NULL, "survival:cox", NULL)
  expect_equal(
    res_cox$dmatrix_args$label,
    ifelse(cancer$status == 2, cancer$time, -cancer$time)
  )
  expect_equal(
    res_cox$params$objective,
    "survival:cox"
  )

  res_aft <- process.y.margin.and.objective(y_right, NULL, "survival:aft", NULL)
  expect_equal(
    res_aft$dmatrix_args$label_lower_bound,
    cancer$time
  )
  expect_equal(
    res_aft$dmatrix_args$label_upper_bound,
    ifelse(cancer$status == 2, cancer$time, Inf)
  )
  expect_equal(
    res_aft$params$objective,
    "survival:aft"
  )

  y_left <- Surv(seq(1, 4), c(1, 0, 1, 0), type = "left")
  expect_error({
    process.y.margin.and.objective(y_left, NULL, "survival:cox", NULL)
  })
  res_aft <- process.y.margin.and.objective(y_left, NULL, "survival:aft", NULL)
  expect_equal(
    res_aft$dmatrix_args$label_lower_bound,
    c(1, 0, 3, 0)
  )
  expect_equal(
    res_aft$dmatrix_args$label_upper_bound,
    seq(1, 4)
  )
  expect_equal(
    res_aft$params$objective,
    "survival:aft"
  )

  y_interval <- Surv(
    time = c(1, 5, 2, 10, 3),
    time2 = c(2, 5, 2.5, 10, 3),
    event = c(3, 1, 3, 0, 2),
    type = "interval"
  )
  expect_error({
    process.y.margin.and.objective(y_interval, NULL, "survival:cox", NULL)
  })
  res_aft <- process.y.margin.and.objective(y_interval, NULL, "survival:aft", NULL)
  expect_equal(
    res_aft$dmatrix_args$label_lower_bound,
    c(1, 5, 2, 10, 0)
  )
  expect_equal(
    res_aft$dmatrix_args$label_upper_bound,
    c(2, 5, 2.5, Inf, 3)
  )
  expect_equal(
    res_aft$params$objective,
    "survival:aft"
  )

  y_interval_neg <- Surv(
    time = c(1, -5, 2, 10, 3),
    time2 = c(2, -5, 2.5, 10, 3),
    event = c(3, 1, 3, 0, 2),
    type = "interval"
  )
  expect_error({
    process.y.margin.and.objective(y_interval_neg, NULL, "survival:aft", NULL)
  })
})

test_that("Process multi-target", {
  data(mtcars)
  y_multi <- data.frame(
    y1 = mtcars$mpg,
    y2 = mtcars$mpg ^ 2
  )
  for (y_inp in list(y_multi, as.matrix(y_multi), data.table::as.data.table(y_multi))) {
    res_multi <- process.y.margin.and.objective(y_inp, NULL, "reg:pseudohubererror", NULL)
    expect_equal(
      res_multi$dmatrix_args$label,
      as.matrix(y_multi)
    )
    expect_equal(
      res_multi$metadata$y_names,
      c("y1", "y2")
    )
    expect_equal(
      res_multi$params$objective,
      "reg:pseudohubererror"
    )
  }

  expect_error({
    process.y.margin.and.objective(y_multi, NULL, "count:poisson", NULL)
  })

  y_bad <- data.frame(
    c1 = seq(1, 3),
    c2 = rep(as.Date("2024-01-01"), 3)
  )
  expect_error({
    process.y.margin.and.objective(y_bad, NULL, "reg:squarederror", NULL)
  })

  y_bad <- data.frame(
    c1 = seq(1, 3),
    c2 = factor(c('a', 'b', 'a'), c('a', 'b'))
  )
  expect_error({
    process.y.margin.and.objective(y_bad, NULL, "reg:squarederror", NULL)
  })

  y_bad <- seq(1, 20)
  dim(y_bad) <- c(5, 2, 2)
  expect_error({
    process.y.margin.and.objective(y_bad, NULL, "reg:squarederror", NULL)
  })
})

test_that("Process base_margin", {
  y <- seq(101, 110)
  bm_good <- seq(1, 10)
  for (bm in list(bm_good, as.matrix(bm_good), as.data.frame(as.matrix(bm_good)))) {
    res <- process.y.margin.and.objective(y, bm, "reg:squarederror", NULL)
    expect_equal(
      res$dmatrix_args$base_margin,
      seq(1, 10)
    )
  }
  expect_error({
    process.y.margin.and.objective(y, 5, "reg:squarederror", NULL)
  })
  expect_error({
    process.y.margin.and.objective(y, seq(1, 5), "reg:squarederror", NULL)
  })
  expect_error({
    process.y.margin.and.objective(y, matrix(seq(1, 20), ncol = 2), "reg:squarederror", NULL)
  })
  expect_error({
    process.y.margin.and.objective(
      y,
      as.data.frame(matrix(seq(1, 20), ncol = 2)),
      "reg:squarederror",
      NULL
    )
  })

  y <- factor(c('a', 'b', 'c', 'a'))
  bm_good <- matrix(seq(1, 12), ncol = 3)
  for (bm in list(bm_good, as.data.frame(bm_good))) {
    res <- process.y.margin.and.objective(y, bm, "multi:softprob", NULL)
    expect_equal(
      res$dmatrix_args$base_margin |> unname(),
      matrix(seq(1, 12), ncol = 3)
    )
  }
  expect_error({
    process.y.margin.and.objective(y, as.numeric(bm_good), "multi:softprob", NULL)
  })
  expect_error({
    process.y.margin.and.objective(y, 5, "multi:softprob", NULL)
  })
  expect_error({
    process.y.margin.and.objective(y, bm_good[, 1], "multi:softprob", NULL)
  })
  expect_error({
    process.y.margin.and.objective(y, bm_good[, c(1, 2)], "multi:softprob", NULL)
  })
  expect_error({
    process.y.margin.and.objective(y, bm_good[c(1, 2), ], "multi:softprob", NULL)
  })

  y <- seq(101, 110)
  bm_good <- matrix(seq(1, 30), ncol = 3)
  params <- list(quantile_alpha = c(0.1, 0.5, 0.9))
  for (bm in list(bm_good, as.data.frame(bm_good))) {
    res <- process.y.margin.and.objective(y, bm, "reg:quantileerror", params)
    expect_equal(
      res$dmatrix_args$base_margin |> unname(),
      matrix(seq(1, 30), ncol = 3)
    )
  }
  expect_error({
    process.y.margin.and.objective(y, as.numeric(bm_good), "reg:quantileerror", params)
  })
  expect_error({
    process.y.margin.and.objective(y, 5, "reg:quantileerror", params)
  })
  expect_error({
    process.y.margin.and.objective(y, bm_good[, 1], "reg:quantileerror", params)
  })
  expect_error({
    process.y.margin.and.objective(y, bm_good[, c(1, 2)], "reg:quantileerror", params)
  })
  expect_error({
    process.y.margin.and.objective(y, bm_good[c(1, 2, 3), ], "reg:quantileerror", params)
  })

  y <- matrix(seq(101, 130), ncol = 3)
  for (bm in list(bm_good, as.data.frame(bm_good))) {
    res <- process.y.margin.and.objective(y, bm, "reg:squarederror", params)
    expect_equal(
      res$dmatrix_args$base_margin |> unname(),
      matrix(seq(1, 30), ncol = 3)
    )
  }
  expect_error({
    process.y.margin.and.objective(y, as.numeric(bm_good), "reg:squarederror", params)
  })
  expect_error({
    process.y.margin.and.objective(y, 5, "reg:squarederror", params)
  })
  expect_error({
    process.y.margin.and.objective(y, bm_good[, 1], "reg:squarederror", params)
  })
  expect_error({
    process.y.margin.and.objective(y, bm_good[, c(1, 2)], "reg:squarederror", params)
  })
  expect_error({
    process.y.margin.and.objective(y, bm_good[c(1, 2, 3), ], "reg:squarederror", params)
  })
})

test_that("Process monotone constraints", {
  data(iris)
  mc_list <- list(Sepal.Width = 1)
  res <- process.x.and.col.args(
    iris,
    monotone_constraints = mc_list,
    interaction_constraints = NULL,
    feature_weights = NULL,
    lst_args = list(),
    use_qdm = FALSE
  )
  expect_equal(
    res$params$monotone_constraints,
    c(0, 1, 0, 0, 0)
  )

  mc_list2 <- list(Sepal.Width = 1, Petal.Width = -1)
  res <- process.x.and.col.args(
    iris,
    monotone_constraints = mc_list2,
    interaction_constraints = NULL,
    feature_weights = NULL,
    lst_args = list(),
    use_qdm = FALSE
  )
  expect_equal(
    res$params$monotone_constraints,
    c(0, 1, 0, -1, 0)
  )

  mc_vec <- c(0, 1, -1, 0, 0)
  res <- process.x.and.col.args(
    iris,
    monotone_constraints = mc_vec,
    interaction_constraints = NULL,
    feature_weights = NULL,
    lst_args = list(),
    use_qdm = FALSE
  )
  expect_equal(
    res$params$monotone_constraints,
    c(0, 1, -1, 0, 0)
  )

  mc_named_vec <- c(1, 1)
  names(mc_named_vec) <- names(iris)[1:2]
  res <- process.x.and.col.args(
    iris,
    monotone_constraints = mc_named_vec,
    interaction_constraints = NULL,
    feature_weights = NULL,
    lst_args = list(),
    use_qdm = FALSE
  )
  expect_equal(
    res$params$monotone_constraints,
    c(1, 1, 0, 0, 0)
  )

  mc_named_all <- c(0, -1, 1, 0, -1)
  names(mc_named_all) <- rev(names(iris))
  res <- process.x.and.col.args(
    iris,
    monotone_constraints = mc_named_all,
    interaction_constraints = NULL,
    feature_weights = NULL,
    lst_args = list(),
    use_qdm = FALSE
  )
  expect_equal(
    res$params$monotone_constraints,
    rev(mc_named_all) |> unname()
  )

  expect_error({
    process.x.and.col.args(
      iris,
      monotone_constraints = list(
        Sepal.Width = 1,
        Petal.Width = -1,
        Sepal.Width = -1
      ),
      interaction_constraints = NULL,
      feature_weights = NULL,
      lst_args = list(),
      use_qdm = FALSE
    )
  })

  expect_error({
    process.x.and.col.args(
      iris,
      monotone_constraints = rep(0, 6),
      interaction_constraints = NULL,
      feature_weights = NULL,
      lst_args = list(),
      use_qdm = FALSE
    )
  })
})

test_that("Process interaction_constraints", {
  data(iris)
  res <- process.x.and.col.args(iris, NULL, list(c(1L, 2L)), NULL, NULL, FALSE)
  expect_equal(
    res$params$interaction_constraints,
    list(c(0, 1))
  )
  res <- process.x.and.col.args(iris, NULL, list(c(1.0, 2.0)), NULL, NULL, FALSE)
  expect_equal(
    res$params$interaction_constraints,
    list(c(0, 1))
  )
  res <- process.x.and.col.args(iris, NULL, list(c(1, 2), c(3, 4)), NULL, NULL, FALSE)
  expect_equal(
    res$params$interaction_constraints,
    list(c(0, 1), c(2, 3))
  )
  res <- process.x.and.col.args(
    iris, NULL, list(c("Sepal.Length", "Sepal.Width")), NULL, NULL, FALSE
  )
  expect_equal(
    res$params$interaction_constraints,
    list(c(0, 1))
  )
  res <- process.x.and.col.args(
    as.matrix(iris),
    NULL,
    list(c("Sepal.Length", "Sepal.Width")),
    NULL,
    NULL,
    FALSE
  )
  expect_equal(
    res$params$interaction_constraints,
    list(c(0, 1))
  )
  res <- process.x.and.col.args(
    iris,
    NULL,
    list(c("Sepal.Width", "Petal.Length"), c("Sepal.Length", "Petal.Width", "Species")),
    NULL,
    NULL,
    FALSE
  )
  expect_equal(
    res$params$interaction_constraints,
    list(c(1, 2), c(0, 3, 4))
  )

  expect_error({
    process.x.and.col.args(iris, NULL, list(c(1L, 20L)), NULL, NULL, FALSE)
  })
  expect_error({
    process.x.and.col.args(iris, NULL, list(c(0L, 2L)), NULL, NULL, FALSE)
  })
  expect_error({
    process.x.and.col.args(iris, NULL, list(c("1", "2")), NULL, NULL, FALSE)
  })
  expect_error({
    process.x.and.col.args(iris, NULL, list(c("Sepal", "Petal")), NULL, NULL, FALSE)
  })
  expect_error({
    process.x.and.col.args(iris, NULL, c(1L, 2L), NULL, NULL, FALSE)
  })
  expect_error({
    process.x.and.col.args(iris, NULL, matrix(c(1L, 2L)), NULL, NULL, FALSE)
  })
  expect_error({
    process.x.and.col.args(iris, NULL, list(c(1, 2.5)), NULL, NULL, FALSE)
  })
})

test_that("Sparse matrices are casted to CSR for QDM", {
  data(agaricus.test, package = "xgboost")
  x <- agaricus.test$data
  for (x_in in list(x, methods::as(x, "TsparseMatrix"))) {
    res <- process.x.and.col.args(
      x_in,
      NULL,
      NULL,
      NULL,
      NULL,
      TRUE
    )
    expect_s4_class(res$dmatrix_args$data, "dgRMatrix")
  }
})

test_that("Process feature_weights", {
  data(iris)
  w_vector <- seq(1, 5)
  res <-  process.x.and.col.args(
    iris,
    monotone_constraints = NULL,
    interaction_constraints = NULL,
    feature_weights = w_vector,
    lst_args = list(),
    use_qdm = FALSE
  )
  expect_equal(
    res$dmatrix_args$feature_weights,
    seq(1, 5)
  )

  w_named_vector <- seq(1, 5)
  names(w_named_vector) <- rev(names(iris))
  res <-  process.x.and.col.args(
    iris,
    monotone_constraints = NULL,
    interaction_constraints = NULL,
    feature_weights = w_named_vector,
    lst_args = list(),
    use_qdm = FALSE
  )
  expect_equal(
    res$dmatrix_args$feature_weights,
    rev(seq(1, 5))
  )

  w_list <- list(
    Species = 5,
    Sepal.Length = 1,
    Sepal.Width = 2,
    Petal.Length = 3,
    Petal.Width = 4
  )
  res <- process.x.and.col.args(
    iris,
    monotone_constraints = NULL,
    interaction_constraints = NULL,
    feature_weights = w_list,
    lst_args = list(),
    use_qdm = FALSE
  )
  expect_equal(
    res$dmatrix_args$feature_weights,
    seq(1, 5)
  )
})

test_that("Whole function works", {
  data(cancer, package = "survival")
  y <- Surv(cancer$time, cancer$status - 1, type = "right")
  x <- as.data.table(cancer)[, -c("time", "status")]
  model <- xgboost(
    x,
    y,
    monotone_constraints = list(age = -1),
    nthreads = 1L,
    nrounds = 5L,
    learning_rate = 3
  )
  expect_equal(
    attributes(model)$params$objective,
    "survival:aft"
  )
  expect_equal(
    attributes(model)$metadata$n_targets,
    1L
  )
  expect_equal(
    attributes(model)$params$monotone_constraints,
    "(0,-1,0,0,0,0,0,0)"
  )
  expect_false(
    "interaction_constraints" %in% names(attributes(model)$params)
  )
  expect_equal(
    attributes(model)$params$learning_rate,
    3
  )
  txt <- capture.output({
    print(model)
  })
  expect_true(any(grepl("Objective: survival:aft", txt, fixed = TRUE)))
  expect_true(any(grepl("monotone_constraints", txt, fixed = TRUE)))
  expect_true(any(grepl("Number of iterations: 5", txt, fixed = TRUE)))
  expect_true(any(grepl("Number of features: 8", txt, fixed = TRUE)))
})

test_that("Can predict probabilities and raw scores", {
  y <- ToothGrowth$supp
  x <- ToothGrowth[, -2L]
  model <- xgboost(x, y, nthreads = 1L, nrounds = 3L, max_depth = 2L)
  pred_prob <- predict(model, x, type = "response")
  pred_raw <- predict(model, x, type = "raw")
  expect_true(is.vector(pred_prob))
  expect_equal(length(pred_prob), nrow(x))
  expect_true(min(pred_prob) >= 0)
  expect_true(max(pred_prob) <= 1)

  expect_equal(length(pred_raw), nrow(x))
  expect_true(is.vector(pred_raw))
  expect_true(min(pred_raw) < 0)
  expect_true(max(pred_raw) > 0)

  expect_equal(
    pred_prob,
    1 / (1 + exp(-pred_raw)),
    tolerance = 1e-6
  )
})

test_that("Can predict class", {
  y <- iris$Species
  x <- iris[, -5L]
  model <- xgboost(x, y, nthreads = 1L, nrounds = 3L, max_depth = 2L)
  pred_class <- predict(model, x, type = "class")
  expect_true(is.factor(pred_class))
  expect_equal(levels(pred_class), levels(y))

  y <- ToothGrowth$supp
  x <- ToothGrowth[, -2L]
  model <- xgboost(x, y, nthreads = 1L, nrounds = 3L, max_depth = 2L)
  pred_class <- predict(model, x, type = "class")
  expect_true(is.factor(pred_class))
  expect_equal(levels(pred_class), levels(y))

  probs <- predict(model, x, type = "response")
  expect_true(all(pred_class[probs >= 0.5] == levels(y)[[2L]]))
  expect_true(all(pred_class[probs < 0.5] == levels(y)[[1L]]))

  # Check that it fails for regression models
  y <- mtcars$mpg
  x <- mtcars[, -1L]
  model <- xgboost(x, y, nthreads = 1L, nrounds = 3L, max_depth = 2L)
  expect_error({
    predict(model, x, type = "class")
  })
})

test_that("Metadata survives serialization", {
  y <- iris$Species
  x <- iris[, -5L]
  model_fresh <- xgboost(x, y, nthreads = 1L, nrounds = 3L, max_depth = 2L)
  temp_file <- file.path(tempdir(), "xgb_model.Rds")
  saveRDS(model_fresh, temp_file)
  model <- readRDS(temp_file)
  pred_class <- predict(model, x, type = "class")
  expect_true(is.factor(pred_class))
  expect_equal(levels(pred_class), levels(y))
})

test_that("Column names aren't added when not appropriate", {
  pred_types <- c(
    "response",
    "raw",
    "leaf"
  )
  for (pred_type in pred_types) {
    y <- mtcars$mpg
    x <- mtcars[, -1L]
    model <- xgboost(
      x,
      y,
      nthreads = 1L,
      nrounds = 3L,
      max_depth = 2L,
      objective = "reg:quantileerror",
      quantile_alpha = 0.5
    )
    pred <- predict(model, x, type = pred_type)
    if (pred_type %in% c("raw", "response")) {
      expect_true(is.vector(pred))
    } else {
      expect_true(length(dim(pred)) >= 2L)
      expect_null(colnames(pred))
    }

    y <- ToothGrowth$supp
    x <- ToothGrowth[, -2L]
    model <- xgboost(x, y, nthreads = 1L, nrounds = 3L, max_depth = 2L)
    pred <- predict(model, x, type = pred_type)
    if (pred_type %in% c("raw", "response")) {
      expect_true(is.vector(pred))
    } else {
      expect_true(length(dim(pred)) >= 2L)
      expect_null(colnames(pred))
    }
  }
})

test_that("Column names from multiclass are added to non-class predictions", {
  y <- iris$Species
  x <- iris[, -5L]
  model <- xgboost(x, y, nthreads = 1L, nrounds = 3L, max_depth = 2L)

  pred_types_with_colnames <- c(
    "response",
    "raw",
    "contrib",
    "interaction"
  )

  for (pred_type in pred_types_with_colnames) {
    pred <- predict(model, x, type = pred_type)
    expect_equal(nrow(pred), nrow(x))
    expect_equal(ncol(pred), 3L)
    expect_equal(colnames(pred), levels(y))
  }
})

test_that("Column names from multitarget are added to predictions", {
  y <- data.frame(
    ylog = log(mtcars$mpg),
    ysqrt = sqrt(mtcars$mpg)
  )
  x <- mtcars[, -1L]
  model <- xgboost(x, y, nthreads = 1L, nrounds = 3L, max_depth = 2L)

  pred_types_with_colnames <- c(
    "response",
    "raw",
    "contrib",
    "interaction"
  )

  for (pred_type in pred_types_with_colnames) {
    pred <- predict(model, x, type = pred_type)
    expect_equal(nrow(pred), nrow(x))
    expect_equal(ncol(pred), 2L)
    expect_equal(colnames(pred), colnames(y))
  }
})

test_that("Column names from multiquantile are added to predictions", {
  y <- mtcars$mpg
  x <- mtcars[, -1L]
  model <- xgboost(
    x,
    y,
    nthreads = 1L,
    nrounds = 3L,
    max_depth = 2L,
    objective = "reg:quantileerror",
    quantile_alpha = c(0.25, 0.5, 0.75)
  )

  pred_types_with_colnames <- c(
    "response",
    "raw",
    "contrib",
    "interaction"
  )

  for (pred_type in pred_types_with_colnames) {
    pred <- predict(model, x, type = pred_type)
    expect_equal(nrow(pred), nrow(x))
    expect_equal(ncol(pred), 3L)
    expect_equal(colnames(pred), c("q0.25", "q0.5", "q0.75"))
  }
})

test_that("Leaf predictions have multiple dimensions when needed", {
  # single score, multiple trees
  y <- mtcars$mpg
  x <- mtcars[, -1L]
  model <- xgboost(
    x,
    y,
    nthreads = 1L,
    nrounds = 4L,
    max_depth = 2L,
    objective = "reg:quantileerror",
    quantile_alpha = 0.5
  )
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 4L))
  expect_equal(row.names(pred), row.names(x))
  expect_null(colnames(pred))

  # single score, single tree
  model <- xgboost(
    x,
    y,
    nthreads = 1L,
    nrounds = 1L,
    max_depth = 2L,
    objective = "reg:quantileerror",
    quantile_alpha = 0.5
  )
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 1L))
  expect_equal(row.names(pred), row.names(x))
  expect_null(colnames(pred))

  # multiple score, multiple trees
  model <- xgboost(
    x,
    y,
    nthreads = 1L,
    nrounds = 4L,
    max_depth = 2L,
    objective = "reg:quantileerror",
    quantile_alpha = c(0.25, 0.5, 0.75)
  )
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 4L, 3L))
  expect_equal(row.names(pred), row.names(x))
  expect_null(colnames(pred))
  expect_equal(dimnames(pred)[[3L]], c("q0.25", "q0.5", "q0.75"))

  # multiple score, single tree
  model <- xgboost(
    x,
    y,
    nthreads = 1L,
    nrounds = 1L,
    max_depth = 2L,
    objective = "reg:quantileerror",
    quantile_alpha = c(0.25, 0.5, 0.75)
  )
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 1L, 3L))
  expect_equal(row.names(pred), row.names(x))
  expect_null(colnames(pred))
  expect_equal(dimnames(pred)[[3L]], c("q0.25", "q0.5", "q0.75"))

  # parallel trees, single tree, single score
  model <- xgboost(
    x,
    y,
    nthreads = 1L,
    nrounds = 1L,
    max_depth = 2L,
    objective = "count:poisson",
    num_parallel_tree = 2L
  )
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 1L, 2L))
  expect_equal(row.names(pred), row.names(x))
  expect_null(colnames(pred))
  expect_null(dimnames(pred)[[3L]])

  # num_parallel_tree>1 + multiple scores is not supported at the moment so no test for it.
})

test_that("Column names from multiclass are added to leaf predictions", {
  y <- iris$Species
  x <- iris[, -5L]
  model <- xgboost(x, y, nthreads = 1L, nrounds = 4L, max_depth = 2L)
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 4L, 3L))
  expect_equal(dimnames(pred)[[3L]], levels(y))

  # Check also for a single tree
  model <- xgboost(x, y, nthreads = 1L, nrounds = 1L, max_depth = 2L)
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 1L, 3L))
  expect_equal(dimnames(pred)[[3L]], levels(y))
})

test_that("Column names from multitarget are added to leaf predictions", {
  y <- data.frame(
    ylog = log(mtcars$mpg),
    ysqrt = sqrt(mtcars$mpg)
  )
  x <- mtcars[, -1L]
  model <- xgboost(x, y, nthreads = 1L, nrounds = 4L, max_depth = 2L)
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 4L, 2L))
  expect_equal(dimnames(pred)[[3L]], colnames(y))

  # Check also for a single tree
  model <- xgboost(x, y, nthreads = 1L, nrounds = 1L, max_depth = 2L)
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 1L, 2L))
  expect_equal(dimnames(pred)[[3L]], colnames(y))
})

test_that("Column names from multiquantile are added to leaf predictions", {
  y <- mtcars$mpg
  x <- mtcars[, -1L]
  model <- xgboost(
    x,
    y,
    nthreads = 1L,
    nrounds = 4L,
    max_depth = 2L,
    objective = "reg:quantileerror",
    quantile_alpha = c(0.25, 0.5, 0.75)
  )
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 4L, 3L))
  expect_equal(dimnames(pred)[[3L]], c("q0.25", "q0.5", "q0.75"))

  # Check also for a single tree
  model <- xgboost(
    x,
    y,
    nthreads = 1L,
    nrounds = 1L,
    max_depth = 2L,
    objective = "reg:quantileerror",
    quantile_alpha = c(0.25, 0.5, 0.75)
  )
  pred <- predict(model, x, type = "leaf")
  expect_equal(dim(pred), c(nrow(x), 1L, 3L))
  expect_equal(dimnames(pred)[[3L]], c("q0.25", "q0.5", "q0.75"))
})

test_that("Evaluation fraction leaves examples of all classes for training", {
  # With minimal sample leave no remainder
  lst_args <- list(
    dmatrix_args = list(
      data = matrix(seq(1, 4), ncol = 1L),
      label = c(0, 0, 1, 1)
    ),
    metadata = list(
      y_levels = c("a", "b")
    ),
    params = list(
      seed = 123
    )
  )
  for (retry in seq_len(10)) {
    lst_args$params$seed <- retry
    res <- process.eval.set(0.5, lst_args)
    expect_equal(length(intersect(res$idx_train, res$idx_eval)), 0)
    expect_equal(length(res$idx_train), 2L)
    expect_equal(length(res$idx_eval), 2L)
    expect_true(length(intersect(c(1L, 2L), res$idx_train)) >= 1L)
    expect_true(length(intersect(c(3L, 4L), res$idx_train)) >= 1L)
  }

  # With minimal sample leaving some remainder
  lst_args <- list(
    dmatrix_args = list(
      data = matrix(seq(1, 5), ncol = 1L),
      label = c(0, 0, 1, 1, 1)
    ),
    metadata = list(
      y_levels = c("a", "b")
    ),
    params = list(
      seed = 123
    )
  )
  for (retry in seq_len(20)) {
    lst_args$params$seed <- retry
    res <- process.eval.set(0.4, lst_args)
    expect_equal(length(intersect(res$idx_train, res$idx_eval)), 0)
    expect_equal(length(res$idx_train), 3L)
    expect_equal(length(res$idx_eval), 2L)
    expect_true(length(intersect(c(1L, 2L), res$idx_train)) >= 1L)
    expect_true(length(intersect(c(3L, 4L, 5L), res$idx_train)) >= 1L)
  }
})

test_that("'eval_set' as fraction works", {
  y <- iris$Species
  x <- iris[, -5L]
  model <- xgboost(
    x,
    y,
    base_margin = matrix(0.1, nrow = nrow(x), ncol = 3L),
    eval_set = 0.2,
    nthreads = 1L,
    nrounds = 4L,
    max_depth = 2L,
    verbosity = 0L
  )
  expect_true(hasName(attributes(model), "evaluation_log"))
  evaluation_log <- attributes(model)$evaluation_log
  expect_equal(nrow(evaluation_log), 4L)
  expect_true(hasName(evaluation_log, "eval_mlogloss"))
  expect_equal(length(attributes(model)$metadata$y_levels), 3L)
})

test_that("Linear booster importance uses class names", {
  y <- iris$Species
  x <- iris[, -5L]
  model <- xgboost(
    x,
    y,
    nthreads = 1L,
    nrounds = 4L,
    verbosity = 0L,
    booster = "gblinear",
    learning_rate = 0.2
  )
  imp <- xgb.importance(model)
  expect_true(is.factor(imp$Class))
  expect_equal(levels(imp$Class), levels(y))
})
