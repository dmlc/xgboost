context('Test helper functions')

VCD_AVAILABLE <- requireNamespace("vcd", quietly = TRUE)
.skip_if_vcd_not_available <- function() {
    if (!VCD_AVAILABLE) {
        testthat::skip("Optional testing dependency 'vcd' not found.")
    }
}

float_tolerance <- 5e-6

# disable some tests for 32-bit environment
flag_32bit <- .Machine$sizeof.pointer != 8

set.seed(1982)

nrounds <- 12
if (isTRUE(VCD_AVAILABLE)) {
    data(Arthritis, package = "vcd")
    df <- data.table::data.table(Arthritis, keep.rownames = FALSE)
    df[, AgeDiscret := as.factor(round(Age / 10, 0))]
    df[, AgeCat := as.factor(ifelse(Age > 30, "Old", "Young"))]
    df[, ID := NULL]
    sparse_matrix <- Matrix::sparse.model.matrix(Improved~.-1, data = df) # nolint
    label <- df[, ifelse(Improved == "Marked", 1, 0)]

    # binary
    bst.Tree <- xgb.train(
      data = xgb.DMatrix(sparse_matrix, label = label),
      nrounds = nrounds, verbose = 0,
      params = xgb.params(
        max_depth = 9,
        learning_rate = 1,
        nthread = 2,
        objective = "binary:logistic",
        booster = "gbtree",
        base_score = 0.5
      )
    )

    bst.GLM <- xgb.train(
      data = xgb.DMatrix(sparse_matrix, label = label),
      nrounds = nrounds, verbose = 0,
      params = xgb.params(
        learning_rate = 1,
        nthread = 1,
        objective = "binary:logistic",
        booster = "gblinear",
        base_score = 0.5
      )
    )

    feature.names <- colnames(sparse_matrix)

    # without feature names
    bst.Tree.unnamed <- xgb.copy.Booster(bst.Tree)
    setinfo(bst.Tree.unnamed, "feature_name", NULL)
}

# multiclass
mlabel <- as.numeric(iris$Species) - 1
nclass <- 3
mbst.Tree <- xgb.train(
  data = xgb.DMatrix(as.matrix(iris[, -5]), label = mlabel),
  verbose = 0,
  nrounds = nrounds,
  params = xgb.params(
    max_depth = 3, learning_rate = 0.5, nthread = 2,
    objective = "multi:softprob", num_class = nclass, base_score = 0
  )
)

mbst.GLM <- xgb.train(
  data = xgb.DMatrix(as.matrix(iris[, -5]), label = mlabel),
  verbose = 0,
  nrounds = nrounds,
  params = xgb.params(
    booster = "gblinear", learning_rate = 0.1, nthread = 1,
    objective = "multi:softprob", num_class = nclass, base_score = 0
  )
)

test_that("xgb.dump works", {
  .skip_if_vcd_not_available()
  if (!flag_32bit)
    expect_length(xgb.dump(bst.Tree), 200)
  dump_file <- file.path(tempdir(), 'xgb.model.dump')
  expect_true(xgb.dump(bst.Tree, dump_file, with_stats = TRUE))
  expect_true(file.exists(dump_file))
  expect_gt(file.size(dump_file), 8000)

  # JSON format
  dmp <- xgb.dump(bst.Tree, dump_format = "json")
  expect_length(dmp, 1)
  if (!flag_32bit)
    expect_length(grep('nodeid', strsplit(dmp, '\n', fixed = TRUE)[[1]], fixed = TRUE), 188)
})

test_that("xgb.dump works for gblinear", {
  .skip_if_vcd_not_available()
  expect_length(xgb.dump(bst.GLM), 14)
  # also make sure that it works properly for a sparse model where some coefficients
  # are 0 from setting large L1 regularization:
  bst.GLM.sp <- xgb.train(
    data = xgb.DMatrix(sparse_matrix, label = label),
    nrounds = 1,
    params = xgb.params(
      learning_rate = 1,
      nthread = 2,
      reg_alpha = 2,
      objective = "binary:logistic",
      booster = "gblinear"
    )
  )
  d.sp <- xgb.dump(bst.GLM.sp)
  expect_length(d.sp, 14)
  expect_gt(sum(d.sp == "0"), 0)

  # JSON format
  dmp <- xgb.dump(bst.GLM.sp, dump_format = "json")
  expect_length(dmp, 1)
  expect_length(grep('\\d', strsplit(dmp, '\n', fixed = TRUE)[[1]]), 11)
})

test_that("predict leafs works", {
  .skip_if_vcd_not_available()
  # no error for gbtree
  expect_error(pred_leaf <- predict(bst.Tree, sparse_matrix, predleaf = TRUE), regexp = NA)
  expect_equal(dim(pred_leaf), c(nrow(sparse_matrix), nrounds))
  # error for gblinear
  expect_error(predict(bst.GLM, sparse_matrix, predleaf = TRUE))
})

test_that("predict feature contributions works", {
  .skip_if_vcd_not_available()
  # gbtree binary classifier
  expect_error(pred_contr <- predict(bst.Tree, sparse_matrix, predcontrib = TRUE), regexp = NA)
  expect_equal(dim(pred_contr), c(nrow(sparse_matrix), ncol(sparse_matrix) + 1))
  expect_equal(colnames(pred_contr), c(colnames(sparse_matrix), "(Intercept)"))
  pred <- predict(bst.Tree, sparse_matrix, outputmargin = TRUE)
  expect_lt(max(abs(rowSums(pred_contr) - pred)), 1e-5)
  # must work with data that has no column names
  X <- sparse_matrix
  colnames(X) <- NULL
  expect_error(pred_contr_ <- predict(bst.Tree, X, predcontrib = TRUE), regexp = NA)
  expect_equal(pred_contr, pred_contr_, check.attributes = FALSE,
               tolerance = float_tolerance)

  # gbtree binary classifier (approximate method)
  expect_error(pred_contr <- predict(bst.Tree, sparse_matrix, predcontrib = TRUE, approxcontrib = TRUE), regexp = NA)
  expect_equal(dim(pred_contr), c(nrow(sparse_matrix), ncol(sparse_matrix) + 1))
  expect_equal(colnames(pred_contr), c(colnames(sparse_matrix), "(Intercept)"))
  pred <- predict(bst.Tree, sparse_matrix, outputmargin = TRUE)
  expect_lt(max(abs(rowSums(pred_contr) - pred)), 1e-5)

  # gblinear binary classifier
  expect_error(pred_contr <- predict(bst.GLM, sparse_matrix, predcontrib = TRUE), regexp = NA)
  expect_equal(dim(pred_contr), c(nrow(sparse_matrix), ncol(sparse_matrix) + 1))
  expect_equal(colnames(pred_contr), c(colnames(sparse_matrix), "(Intercept)"))
  pred <- predict(bst.GLM, sparse_matrix, outputmargin = TRUE)
  expect_lt(max(abs(rowSums(pred_contr) - pred)), 1e-5)
  # manual calculation of linear terms
  coefs <- as.numeric(xgb.dump(bst.GLM)[-c(1, 2, 4)])
  coefs <- c(coefs[-1], coefs[1]) # intercept must be the last
  pred_contr_manual <- sweep(cbind(sparse_matrix, 1), 2, coefs, FUN = "*")
  expect_equal(as.numeric(pred_contr), as.numeric(pred_contr_manual),
               tolerance = float_tolerance)

  # gbtree multiclass
  pred <- predict(mbst.Tree, as.matrix(iris[, -5]), outputmargin = TRUE)
  pred_contr <- predict(mbst.Tree, as.matrix(iris[, -5]), predcontrib = TRUE)
  expect_is(pred_contr, "array")
  expect_length(dim(pred_contr), 3)
  for (g in seq_len(dim(pred_contr)[2])) {
    expect_equal(colnames(pred_contr[, g, ]), c(colnames(iris[, -5]), "(Intercept)"))
    expect_lt(max(abs(rowSums(pred_contr[, g, ]) - pred[, g])), 1e-5)
  }

  # gblinear multiclass (set base_score = 0, which is base margin in multiclass)
  pred <- predict(mbst.GLM, as.matrix(iris[, -5]), outputmargin = TRUE)
  pred_contr <- predict(mbst.GLM, as.matrix(iris[, -5]), predcontrib = TRUE)
  expect_length(dim(pred_contr), 3)
  coefs_all <- matrix(
    data = as.numeric(xgb.dump(mbst.GLM)[-c(1, 2, 6)]),
    ncol = 3,
    byrow = TRUE
  )
  for (g in seq_along(dim(pred_contr)[2])) {
    expect_equal(colnames(pred_contr[, g, ]), c(colnames(iris[, -5]), "(Intercept)"))
    expect_lt(max(abs(rowSums(pred_contr[, g, ]) - pred[, g])), float_tolerance)
    # manual calculation of linear terms
    coefs <- c(coefs_all[-1, g], coefs_all[1, g]) # intercept needs to be the last
    pred_contr_manual <- sweep(as.matrix(cbind(iris[, -5], 1)), 2, coefs, FUN = "*")
    expect_equal(as.numeric(pred_contr[, g, ]), as.numeric(pred_contr_manual),
                 tolerance = float_tolerance)
  }
})

test_that("SHAPs sum to predictions, with or without DART", {
  d <- cbind(
    x1 = rnorm(100),
    x2 = rnorm(100),
    x3 = rnorm(100))
  y <- d[, "x1"] + d[, "x2"]^2 +
    ifelse(d[, "x3"] > .5, d[, "x3"]^2, 2^d[, "x3"]) +
    rnorm(100)
  nrounds <- 30

  for (booster in list("gbtree", "dart")) {
    fit <- xgb.train(
      params = c(
        list(
          nthread = 2,
          booster = booster,
          objective = "reg:squarederror",
          eval_metric = "rmse"),
        if (booster == "dart")
          list(rate_drop = .01, one_drop = TRUE)),
      data = xgb.DMatrix(d, label = y),
      nrounds = nrounds)

    pr <- function(...) {
      predict(fit, newdata = d, ...)
    }
    pred <- pr()
    shap <- pr(predcontrib = TRUE)
    shapi <- pr(predinteraction = TRUE)
    tol <- 1e-5

    expect_equal(rowSums(shap), pred, tol = tol)
    expect_equal(rowSums(shapi), pred, tol = tol)
    for (i in seq_len(nrow(d)))
      for (f in list(rowSums, colSums))
        expect_equal(f(shapi[i, , ]), shap[i, ], tol = tol)
  }
})

test_that("xgb-attribute functionality", {
  .skip_if_vcd_not_available()
  val <- "my attribute value"
  list.val <- list(my_attr = val, a = 123, b = 'ok')
  list.ch <- list.val[order(names(list.val))]
  list.ch <- lapply(list.ch, as.character)
  # note: iter is 0-index in xgb attributes
  list.default <- list()
  list.ch <- c(list.ch, list.default)
  # proper input:
  expect_error(xgb.attr(bst.Tree, NULL))
  expect_error(xgb.attr(val, val))
  # set & get:
  expect_null(xgb.attr(bst.Tree, "asdf"))
  expect_equal(xgb.attributes(bst.Tree), list.default)
  bst.Tree.copy <- xgb.copy.Booster(bst.Tree)
  xgb.attr(bst.Tree.copy, "my_attr") <- val
  expect_equal(xgb.attr(bst.Tree.copy, "my_attr"), val)
  xgb.attributes(bst.Tree.copy) <- list.val
  expect_equal(xgb.attributes(bst.Tree.copy), list.ch)
  # serializing:
  fname <- file.path(tempdir(), "xgb.ubj")
  xgb.save(bst.Tree.copy, fname)
  bst <- xgb.load(fname)
  expect_equal(xgb.attr(bst, "my_attr"), val)
  expect_equal(xgb.attributes(bst), list.ch)
  # deletion:
  xgb.attr(bst, "my_attr") <- NULL
  expect_null(xgb.attr(bst, "my_attr"))
  expect_equal(xgb.attributes(bst), list.ch[c("a", "b")])
  xgb.attributes(bst) <- list(a = NULL, b = NULL)
  expect_equal(xgb.attributes(bst), list.default)
  xgb.attributes(bst) <- list(niter = NULL)
  expect_equal(xgb.attributes(bst), list())
})

if (grepl('Windows', Sys.info()[['sysname']], fixed = TRUE) ||
    grepl('Linux', Sys.info()[['sysname']], fixed = TRUE) ||
    grepl('Darwin', Sys.info()[['sysname']], fixed = TRUE)) {
    test_that("xgb-attribute numeric precision", {
      .skip_if_vcd_not_available()
      # check that lossless conversion works with 17 digits
      # numeric -> character -> numeric
      X <- 10^runif(100, -20, 20)
      if (capabilities('long.double')) {
          X2X <- as.numeric(format(X, digits = 17))
          expect_equal(X, X2X, tolerance = float_tolerance)
      }
      # retrieved attributes to be the same as written
      for (x in X) {
        xgb.attr(bst.Tree, "x") <- x
        expect_equal(as.numeric(xgb.attr(bst.Tree, "x")), x, tolerance = float_tolerance)
        xgb.attributes(bst.Tree) <- list(a = "A", b = x)
        expect_equal(as.numeric(xgb.attr(bst.Tree, "b")), x, tolerance = float_tolerance)
      }
    })
}

test_that("xgb.Booster serializing as R object works", {
  .skip_if_vcd_not_available()
  fname_rds <- file.path(tempdir(), "xgb.model.rds")
  saveRDS(bst.Tree, fname_rds)
  bst <- readRDS(fname_rds)
  dtrain <- xgb.DMatrix(sparse_matrix, label = label, nthread = 2)
  expect_equal(predict(bst.Tree, dtrain), predict(bst, dtrain), tolerance = float_tolerance)
  expect_equal(xgb.dump(bst.Tree), xgb.dump(bst))

  fname_bin <- file.path(tempdir(), "xgb.model")
  xgb.save(bst, fname_bin)
  bst <- readRDS(fname_rds)
  expect_equal(predict(bst.Tree, dtrain), predict(bst, dtrain), tolerance = float_tolerance)
})

test_that("xgb.model.dt.tree works with and without feature names", {
  .skip_if_vcd_not_available()
  names.dt.trees <- c("Tree", "Node", "ID", "Feature", "Split", "Yes", "No", "Missing", "Gain", "Cover")
  dt.tree <- xgb.model.dt.tree(model = bst.Tree)
  expect_equal(names.dt.trees, names(dt.tree))
  if (!flag_32bit)
    expect_equal(dim(dt.tree), c(188, 10))
  expect_output(str(dt.tree), 'Feature.*\\"Age\\"')

  # when model contains no feature names:
  dt.tree.x <- xgb.model.dt.tree(model = bst.Tree.unnamed)
  expect_output(str(dt.tree.x), 'Feature.*\\"3\\"')
  expect_equal(dt.tree[, -4, with = FALSE], dt.tree.x[, -4, with = FALSE])

  # using integer node ID instead of character
  dt.tree.int <- xgb.model.dt.tree(model = bst.Tree, use_int_id = TRUE)
  expect_equal(as.integer(data.table::tstrsplit(dt.tree$Yes, '-', fixed = TRUE)[[2]]), dt.tree.int$Yes)
  expect_equal(as.integer(data.table::tstrsplit(dt.tree$No, '-', fixed = TRUE)[[2]]), dt.tree.int$No)
  expect_equal(as.integer(data.table::tstrsplit(dt.tree$Missing, '-', fixed = TRUE)[[2]]), dt.tree.int$Missing)
})

test_that("xgb.model.dt.tree throws error for gblinear", {
  .skip_if_vcd_not_available()
  expect_error(xgb.model.dt.tree(model = bst.GLM))
})

test_that("xgb.importance works with and without feature names", {
  .skip_if_vcd_not_available()
  importance.Tree <- xgb.importance(feature_names = feature.names, model = bst.Tree.unnamed)
  if (!flag_32bit)
    expect_equal(dim(importance.Tree), c(7, 4))
  expect_equal(colnames(importance.Tree), c("Feature", "Gain", "Cover", "Frequency"))
  expect_output(str(importance.Tree), 'Feature.*\\"Age\\"')

  importance.Tree.0 <- xgb.importance(model = bst.Tree)
  expect_equal(importance.Tree, importance.Tree.0, tolerance = float_tolerance)

  # when model contains no feature names:
  importance.Tree.x <- xgb.importance(model = bst.Tree.unnamed)
  expect_equal(importance.Tree[, -1, with = FALSE], importance.Tree.x[, -1, with = FALSE],
               tolerance = float_tolerance)

  imp2plot <- xgb.plot.importance(importance_matrix = importance.Tree)
  expect_equal(colnames(imp2plot), c("Feature", "Gain", "Cover", "Frequency", "Importance"))
  xgb.ggplot.importance(importance_matrix = importance.Tree)

  # for multiclass
  imp.Tree <- xgb.importance(model = mbst.Tree)
  expect_equal(dim(imp.Tree), c(4, 4))

  trees <- seq(from = 1, by = 2, length.out = 2)
  importance <- xgb.importance(feature_names = feature.names, model = bst.Tree, trees = trees)

  importance_from_dump <- function() {
    imp <- xgb.model.dt.tree(
      model = bst.Tree,
      trees = trees
    )[
      Feature != "Leaf", .(
        Gain = sum(Gain),
        Cover = sum(Cover),
        Frequency = .N
      ),
      by = Feature
    ][
      , `:=`(
        Gain = Gain / sum(Gain),
        Cover = Cover / sum(Cover),
        Frequency = Frequency / sum(Frequency)
      )
    ][
      order(Gain, decreasing = TRUE)
    ]
    imp
  }
  expect_equal(importance_from_dump(), importance, tolerance = 1e-6)

  ## decision stump
  m <- xgb.train(
    data = xgb.DMatrix(as.matrix(data.frame(x = c(0, 1))), label = c(1, 2)),
    nrounds = 1,
    params = xgb.params(
      base_score = 0.5,
      nthread = 2
    )
  )
  df <- xgb.model.dt.tree(model = m)
  expect_equal(df$Feature, "Leaf")
  expect_equal(df$Cover, 2)
})

test_that("xgb.importance works with GLM model", {
  .skip_if_vcd_not_available()
  importance.GLM <- xgb.importance(feature_names = feature.names, model = bst.GLM)
  expect_equal(dim(importance.GLM), c(10, 2))
  expect_equal(colnames(importance.GLM), c("Feature", "Weight"))
  xgb.importance(model = bst.GLM)
  imp2plot <- xgb.plot.importance(importance.GLM)
  expect_equal(colnames(imp2plot), c("Feature", "Weight", "Importance"))
  xgb.ggplot.importance(importance.GLM)

  # check that the input is not modified in-place
  expect_false("Importance" %in% names(importance.GLM))

  # for multiclass
  imp.GLM <- xgb.importance(model = mbst.GLM)
  expect_equal(dim(imp.GLM), c(12, 3))
  expect_equal(imp.GLM$Class, rep(0:2, each = 4))
})

test_that("xgb.model.dt.tree and xgb.importance work with a single split model", {
  .skip_if_vcd_not_available()
  bst1 <- xgb.train(
    data = xgb.DMatrix(sparse_matrix, label = label),
    nrounds = 1, verbose = 0,
    params = xgb.params(
      max_depth = 1,
      learning_rate = 1,
      nthread = 2,
      objective = "binary:logistic"
    )
  )
  expect_error(dt <- xgb.model.dt.tree(model = bst1), regexp = NA) # no error
  expect_equal(nrow(dt), 3)
  expect_error(imp <- xgb.importance(model = bst1), regexp = NA) # no error
  expect_equal(nrow(imp), 1)
  expect_equal(imp$Gain, 1)
})

test_that("xgb.plot.importance de-duplicates features", {
  importances <- data.table(
    Feature = c("col1", "col2", "col2"),
    Gain = c(0.4, 0.3, 0.3)
  )
  imp2plot <- xgb.plot.importance(importances)
  expect_equal(nrow(imp2plot), 2L)
  expect_equal(imp2plot$Feature, c("col2", "col1"))
})

test_that("xgb.plot.tree works with and without feature names", {
  .skip_if_vcd_not_available()
  expect_silent(xgb.plot.tree(model = bst.Tree.unnamed))
  expect_silent(xgb.plot.tree(model = bst.Tree))

  ## Categorical
  y <- rnorm(100)
  x <- sample(3, size = 100 * 3, replace = TRUE) |> matrix(nrow = 100)
  x <- x - 1
  dm <- xgb.DMatrix(data = x, label = y)
  setinfo(dm, "feature_type", c("c", "c", "c"))
  model <- xgb.train(
    data = dm,
    params = list(tree_method = "hist"),
    nrounds = 2
  )
  expect_silent(xgb.plot.tree(model = model))
})

test_that("xgb.plot.multi.trees works with and without feature names", {
  .skip_if_vcd_not_available()
  xgb.plot.multi.trees(model = bst.Tree.unnamed, features_keep = 3)
  xgb.plot.multi.trees(model = bst.Tree, features_keep = 3)
  expect_true(TRUE)
})

test_that("xgb.plot.deepness works", {
  .skip_if_vcd_not_available()
  d2p <- xgb.plot.deepness(model = bst.Tree)
  expect_equal(colnames(d2p), c("ID", "Tree", "Depth", "Cover", "Weight"))
  xgb.plot.deepness(model = bst.Tree, which = "med.depth")
  xgb.ggplot.deepness(model = bst.Tree)
})

test_that("xgb.shap.data works when top_n is provided", {
  .skip_if_vcd_not_available()
  data_list <- xgb.shap.data(data = sparse_matrix, model = bst.Tree, top_n = 2)
  expect_equal(names(data_list), c("data", "shap_contrib"))
  expect_equal(NCOL(data_list$data), 2)
  expect_equal(NCOL(data_list$shap_contrib), 2)
  expect_equal(NROW(data_list$data), NROW(data_list$shap_contrib))
  expect_gt(length(colnames(data_list$data)), 0)
  expect_gt(length(colnames(data_list$shap_contrib)), 0)

  # for multiclass without target class provided
  data_list <- xgb.shap.data(data = as.matrix(iris[, -5]), model = mbst.Tree, top_n = 2)
  expect_equal(dim(data_list$shap_contrib), c(nrow(iris), 2))
  # for multiclass with target class provided
  data_list <- xgb.shap.data(data = as.matrix(iris[, -5]), model = mbst.Tree, top_n = 2, target_class = 0)
  expect_equal(dim(data_list$shap_contrib), c(nrow(iris), 2))
})

test_that("xgb.shap.data works with subsampling", {
  .skip_if_vcd_not_available()
  data_list <- xgb.shap.data(data = sparse_matrix, model = bst.Tree, top_n = 2, subsample = 0.8)
  expect_equal(NROW(data_list$data), as.integer(0.8 * nrow(sparse_matrix)))
  expect_equal(NROW(data_list$data), NROW(data_list$shap_contrib))
})

test_that("xgb.shap.data works with data frames", {
  data(mtcars)
  df <- mtcars
  df$cyl <- factor(df$cyl)
  x <- df[, -1]
  y <- df$mpg
  dm <- xgb.DMatrix(x, label = y, nthread = 1L)
  model <- xgb.train(
    data = dm,
    params = list(
      max_depth = 2,
      nthread = 1
    ),
    nrounds = 2
  )
  data_list <- xgb.shap.data(data = df[, -1], model = model, top_n = 2, subsample = 0.8)
  expect_equal(NROW(data_list$data), as.integer(0.8 * nrow(df)))
  expect_equal(NROW(data_list$data), NROW(data_list$shap_contrib))
})

test_that("prepare.ggplot.shap.data works", {
  .skip_if_vcd_not_available()
  data_list <- xgb.shap.data(data = sparse_matrix, model = bst.Tree, top_n = 2)
  plot_data <- prepare.ggplot.shap.data(data_list, normalize = TRUE)
  expect_s3_class(plot_data, "data.frame")
  expect_equal(names(plot_data), c("id", "feature", "feature_value", "shap_value"))
  expect_s3_class(plot_data$feature, "factor")
  # Each observation should have 1 row for each feature
  expect_equal(nrow(plot_data), nrow(sparse_matrix) * 2)
})

test_that("xgb.plot.shap works", {
  .skip_if_vcd_not_available()
  sh <- xgb.plot.shap(data = sparse_matrix, model = bst.Tree, top_n = 2, col = 4)
  expect_equal(names(sh), c("data", "shap_contrib"))
})

test_that("xgb.plot.shap.summary works", {
  .skip_if_vcd_not_available()
  expect_silent(xgb.plot.shap.summary(data = sparse_matrix, model = bst.Tree, top_n = 2))
  expect_silent(xgb.ggplot.shap.summary(data = sparse_matrix, model = bst.Tree, top_n = 2))
})

test_that("xgb.plot.shap.summary ignores categorical features", {
  .skip_if_vcd_not_available()
  data(mtcars)
  df <- mtcars
  df$cyl <- factor(df$cyl)
  levels(df$cyl) <- c("a", "b", "c")
  x <- df[, -1]
  y <- df$mpg
  dm <- xgb.DMatrix(x, label = y, nthread = 1L)
  model <- xgb.train(
    data = dm,
    params = list(
      max_depth = 2,
      nthread = 1
    ),
    nrounds = 2
  )
  expect_warning({
    xgb.ggplot.shap.summary(data = x, model = model, top_n = 2)
  })

  x_num <- mtcars[, -1]
  x_num$gear <- as.numeric(x_num$gear) - 1
  x_num <- as.matrix(x_num)
  dm <- xgb.DMatrix(x_num, label = y, feature_types = c(rep("q", 8), "c", "q"), nthread = 1L)
  model <- xgb.train(
    data = dm,
    params = list(
      max_depth = 2,
      nthread = 1
    ),
    nrounds = 2
  )
  expect_warning({
    xgb.ggplot.shap.summary(data = x_num, model = model, top_n = 2)
  })
})

test_that("check.deprecation works", {
  data(mtcars)
  dm <- xgb.DMatrix(mtcars[, -1L], label = mtcars$mpg)
  params <- xgb.params(nthread = 1, max_depth = 2, eval_metric = "rmse")
  args_train <- list(
    data = dm,
    params = params,
    nrounds = 10,
    verbose = 0
  )

  # with exact name
  options("xgboost.strict_mode" = TRUE)
  expect_error({
    model <- xgb.train(
      data = dm,
      params = params,
      nrounds = 10,
      watchlist = list(tr = dm),
      verbose = 0
    )
  }, regexp = "watchlist")
  options("xgboost.strict_mode" = FALSE)
  expect_warning({
    model <- xgb.train(
      data = dm,
      params = params,
      nrounds = 10,
      watchlist = list(tr = dm),
      verbose = 0
    )
  }, regexp = "watchlist")
  expect_true(hasName(attributes(model), "evaluation_log"))
  expect_equal(names(attributes(model)$evaluation_log), c("iter", "tr_rmse"))

  # with partial name match
  expect_warning({
    model <- xgb.train(
      data = dm,
      params = params,
      nrounds = 10,
      watchlis = list(train = dm),
      verbose = 0
    )
  }, regexp = "watchlist")
  expect_true(hasName(attributes(model), "evaluation_log"))
  expect_equal(names(attributes(model)$evaluation_log), c("iter", "train_rmse"))

  # error/warning is thrown if argument cannot be matched
  options("xgboost.strict_mode" = TRUE)
  expect_error({
    model <- xgb.train(
      data = dm,
      params = params,
      nrounds = 10,
      watchlistt = list(train = dm),
      verbose = 0
    )
  }, regexp = "unrecognized")
  options("xgboost.strict_mode" = FALSE)
  expect_warning({
    model <- xgb.train(
      data = dm,
      params = params,
      nrounds = 10,
      watchlistt = list(train = dm),
      verbose = 0
    )
  }, regexp = "unrecognized")

  # error should suggest to put under 'params' if it goes there
  options("xgboost.strict_mode" = TRUE)
  expect_error({
    model <- xgb.train(
      data = dm,
      nthread = 1, max_depth = 2, eval_metric = "rmse",
      nrounds = 10,
      evals = list(train = dm),
      verbose = 0
    )
  }, regexp = "should be passed as a list to argument 'params'")
  options("xgboost.strict_mode" = FALSE)
  expect_warning({
    model <- xgb.train(
      data = dm,
      nthread = 1, max_depth = 2, eval_metric = "mae",
      nrounds = 10,
      evals = list(train = dm),
      verbose = 0
    )
  }, regexp = "should be passed as a list to argument 'params'")
  expect_true(hasName(attributes(model), "evaluation_log"))
  expect_equal(names(attributes(model)$evaluation_log), c("iter", "train_mae"))

  # can take more than one deprecated parameter
  expect_warning({
    model <- xgb.train(
      training.data = dm,
      params = params,
      nrounds = 10,
      watchlis = list(tr = dm),
      verbose = 0
    )
  }, regexp = "training.data")
  expect_true(hasName(attributes(model), "evaluation_log"))
  expect_equal(names(attributes(model)$evaluation_log), c("iter", "tr_rmse"))
})

test_that('convert.labels works', {
  y <- c(0, 1, 0, 0, 1)
  for (objective in c('binary:logistic', 'binary:logitraw', 'binary:hinge')) {
    res <- xgboost:::convert.labels(y, objective_name = objective)
    expect_s3_class(res, 'factor')
    expect_equal(res, factor(res))
  }
  y <- c(0, 1, 3, 2, 1, 4)
  for (objective in c('multi:softmax', 'multi:softprob', 'rank:pairwise', 'rank:ndcg',
                      'rank:map')) {
    res <- xgboost:::convert.labels(y, objective_name = objective)
    expect_s3_class(res, 'factor')
    expect_equal(res, factor(res))
  }
  y <- c(1.2, 3.0, -1.0, 10.0)
  for (objective in c('reg:squarederror', 'reg:squaredlogerror', 'reg:logistic',
                      'reg:pseudohubererror', 'count:poisson', 'survival:cox', 'survival:aft',
                      'reg:gamma', 'reg:tweedie')) {
    res <- xgboost:::convert.labels(y, objective_name = objective)
    expect_equal(class(res), 'numeric')
  }
})

test_that("validate.features works as expected", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = y, nthread = 1)
  model <- xgb.train(
    params = list(nthread = 1),
    data = dm,
    nrounds = 3
  )

  # result is output as-is when needed
  res <- validate.features(model, x)
  expect_equal(res, x)
  res <- validate.features(model, dm)
  expect_identical(res, dm)
  res <- validate.features(model, as(x[1, ], "dsparseVector"))
  expect_equal(as.numeric(res), unname(x[1, ]))
  res <- validate.features(model, "file.txt")
  expect_equal(res, "file.txt")

  # columns are reordered
  res <- validate.features(model, mtcars[, rev(names(mtcars))])
  expect_equal(names(res), colnames(x))
  expect_equal(as.matrix(res), x)
  res <- validate.features(model, as.matrix(mtcars[, rev(names(mtcars))]))
  expect_equal(colnames(res), colnames(x))
  expect_equal(res, x)
  res <- validate.features(model, mtcars[1, rev(names(mtcars)), drop = FALSE])
  expect_equal(names(res), colnames(x))
  expect_equal(unname(as.matrix(res)), unname(x[1, , drop = FALSE]))
  res <- validate.features(model, as.data.table(mtcars[, rev(names(mtcars))]))
  expect_equal(names(res), colnames(x))
  expect_equal(unname(as.matrix(res)), unname(x))

  # error when columns are missing
  expect_error({
    validate.features(model, mtcars[, 1:3])
  })
  expect_error({
    validate.features(model, as.matrix(mtcars[, 1:ncol(x)])) # nolint
  })
  expect_error({
    validate.features(model, xgb.DMatrix(mtcars[, 1:3]))
  })
  expect_error({
    validate.features(model, as(x[, 1:3], "CsparseMatrix"))
  })

  # error when it cannot reorder or subset
  expect_error({
    validate.features(model, xgb.DMatrix(mtcars))
  }, "Feature names")
  expect_error({
    validate.features(model, xgb.DMatrix(x[, rev(colnames(x))]))
  }, "Feature names")

  # no error about types if the booster doesn't have types
  expect_error({
    validate.features(model, xgb.DMatrix(x, feature_types = c(rep("q", 5), rep("c", 5))))
  }, NA)
  tmp <- mtcars
  tmp[["vs"]] <- factor(tmp[["vs"]])
  expect_error({
    validate.features(model, tmp)
  }, NA)

  # error when types do not match
  setinfo(model, "feature_type", rep("q", 10))
  expect_error({
    validate.features(model, xgb.DMatrix(x, feature_types = c(rep("q", 5), rep("c", 5))))
  }, "Feature types")
  tmp <- mtcars
  tmp[["vs"]] <- factor(tmp[["vs"]])
  expect_error({
    validate.features(model, tmp)
  }, "Feature types")
})

test_that("Parameters constructor works as expected", {
  empty_list <- list()
  names(empty_list) <- character()

  params <- xgb.params()
  expect_equal(params, empty_list)

  params <- xgb.params(max_depth = 2)
  expect_equal(params, list(max_depth = 2))

  params <- xgb.params(max_depth = NULL)
  expect_equal(params, empty_list)

  max_depth <- 3
  params <- xgb.params(max_depth = max_depth)
  expect_equal(params, list(max_depth = 3))

  four <- 4L
  params <- xgb.params(max_depth = four)
  expect_equal(params, list(max_depth = 4L))

  params <- xgb.params(objective = "binary:logistic", nthread = 10)
  expect_equal(params, list(objective = "binary:logistic", nthread = 10))

  expect_error({
    xgb.params(max_xgboost = 10)
  })
  expect_error({
    xgb.params(max_depth = 2, max_depth = 3)
  })
})
