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
    bst.Tree <- xgboost(data = sparse_matrix, label = label, max_depth = 9,
                        eta = 1, nthread = 2, nrounds = nrounds, verbose = 0,
                        objective = "binary:logistic", booster = "gbtree",
                        base_score = 0.5)

    bst.GLM <- xgboost(data = sparse_matrix, label = label,
                       eta = 1, nthread = 1, nrounds = nrounds, verbose = 0,
                       objective = "binary:logistic", booster = "gblinear",
                       base_score = 0.5)

    feature.names <- colnames(sparse_matrix)
}

# multiclass
mlabel <- as.numeric(iris$Species) - 1
nclass <- 3
mbst.Tree <- xgboost(data = as.matrix(iris[, -5]), label = mlabel, verbose = 0,
                     max_depth = 3, eta = 0.5, nthread = 2, nrounds = nrounds,
                     objective = "multi:softprob", num_class = nclass, base_score = 0)

mbst.GLM <- xgboost(data = as.matrix(iris[, -5]), label = mlabel, verbose = 0,
                    booster = "gblinear", eta = 0.1, nthread = 1, nrounds = nrounds,
                    objective = "multi:softprob", num_class = nclass, base_score = 0)


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
  bst.GLM.sp <- xgboost(data = sparse_matrix, label = label, eta = 1, nthread = 2, nrounds = 1,
                        alpha = 2, objective = "binary:logistic", booster = "gblinear")
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
  expect_equal(colnames(pred_contr), c(colnames(sparse_matrix), "BIAS"))
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
  expect_equal(colnames(pred_contr), c(colnames(sparse_matrix), "BIAS"))
  pred <- predict(bst.Tree, sparse_matrix, outputmargin = TRUE)
  expect_lt(max(abs(rowSums(pred_contr) - pred)), 1e-5)

  # gblinear binary classifier
  expect_error(pred_contr <- predict(bst.GLM, sparse_matrix, predcontrib = TRUE), regexp = NA)
  expect_equal(dim(pred_contr), c(nrow(sparse_matrix), ncol(sparse_matrix) + 1))
  expect_equal(colnames(pred_contr), c(colnames(sparse_matrix), "BIAS"))
  pred <- predict(bst.GLM, sparse_matrix, outputmargin = TRUE)
  expect_lt(max(abs(rowSums(pred_contr) - pred)), 1e-5)
  # manual calculation of linear terms
  coefs <- as.numeric(xgb.dump(bst.GLM)[-c(1, 2, 4)])
  coefs <- c(coefs[-1], coefs[1]) # intercept must be the last
  pred_contr_manual <- sweep(cbind(sparse_matrix, 1), 2, coefs, FUN = "*")
  expect_equal(as.numeric(pred_contr), as.numeric(pred_contr_manual),
               tolerance = float_tolerance)

  # gbtree multiclass
  pred <- predict(mbst.Tree, as.matrix(iris[, -5]), outputmargin = TRUE, reshape = TRUE)
  pred_contr <- predict(mbst.Tree, as.matrix(iris[, -5]), predcontrib = TRUE)
  expect_is(pred_contr, "list")
  expect_length(pred_contr, 3)
  for (g in seq_along(pred_contr)) {
    expect_equal(colnames(pred_contr[[g]]), c(colnames(iris[, -5]), "BIAS"))
    expect_lt(max(abs(rowSums(pred_contr[[g]]) - pred[, g])), 1e-5)
  }

  # gblinear multiclass (set base_score = 0, which is base margin in multiclass)
  pred <- predict(mbst.GLM, as.matrix(iris[, -5]), outputmargin = TRUE, reshape = TRUE)
  pred_contr <- predict(mbst.GLM, as.matrix(iris[, -5]), predcontrib = TRUE)
  expect_length(pred_contr, 3)
  coefs_all <- matrix(
    data = as.numeric(xgb.dump(mbst.GLM)[-c(1, 2, 6)]),
    ncol = 3,
    byrow = TRUE
  )
  for (g in seq_along(pred_contr)) {
    expect_equal(colnames(pred_contr[[g]]), c(colnames(iris[, -5]), "BIAS"))
    expect_lt(max(abs(rowSums(pred_contr[[g]]) - pred[, g])), float_tolerance)
    # manual calculation of linear terms
    coefs <- c(coefs_all[-1, g], coefs_all[1, g]) # intercept needs to be the last
    pred_contr_manual <- sweep(as.matrix(cbind(iris[, -5], 1)), 2, coefs, FUN = "*")
    expect_equal(as.numeric(pred_contr[[g]]), as.numeric(pred_contr_manual),
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
    fit <- xgboost(
      params = c(
        list(
          booster = booster,
          objective = "reg:squarederror",
          eval_metric = "rmse"),
        if (booster == "dart")
          list(rate_drop = .01, one_drop = TRUE)),
      data = d,
      label = y,
      nrounds = nrounds)

    pr <- function(...) {
      predict(fit, newdata = d, ...)
    }
    pred <- pr()
    shap <- pr(predcontrib = TRUE)
    shapi <- pr(predinteraction = TRUE)
    tol <- 1e-5

    expect_equal(rowSums(shap), pred, tol = tol)
    expect_equal(apply(shapi, 1, sum), pred, tol = tol)
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
  list.default <- list(niter = as.character(nrounds - 1))
  list.ch <- c(list.ch, list.default)
  # proper input:
  expect_error(xgb.attr(bst.Tree, NULL))
  expect_error(xgb.attr(val, val))
  # set & get:
  expect_null(xgb.attr(bst.Tree, "asdf"))
  expect_equal(xgb.attributes(bst.Tree), list.default)
  xgb.attr(bst.Tree, "my_attr") <- val
  expect_equal(xgb.attr(bst.Tree, "my_attr"), val)
  xgb.attributes(bst.Tree) <- list.val
  expect_equal(xgb.attributes(bst.Tree), list.ch)
  # serializing:
  xgb.save(bst.Tree, 'xgb.model')
  bst <- xgb.load('xgb.model')
  if (file.exists('xgb.model')) file.remove('xgb.model')
  expect_equal(xgb.attr(bst, "my_attr"), val)
  expect_equal(xgb.attributes(bst), list.ch)
  # deletion:
  xgb.attr(bst, "my_attr") <- NULL
  expect_null(xgb.attr(bst, "my_attr"))
  expect_equal(xgb.attributes(bst), list.ch[c("a", "b", "niter")])
  xgb.attributes(bst) <- list(a = NULL, b = NULL)
  expect_equal(xgb.attributes(bst), list.default)
  xgb.attributes(bst) <- list(niter = NULL)
  expect_null(xgb.attributes(bst))
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
  saveRDS(bst.Tree, 'xgb.model.rds')
  bst <- readRDS('xgb.model.rds')
  dtrain <- xgb.DMatrix(sparse_matrix, label = label)
  expect_equal(predict(bst.Tree, dtrain), predict(bst, dtrain), tolerance = float_tolerance)
  expect_equal(xgb.dump(bst.Tree), xgb.dump(bst))
  xgb.save(bst, 'xgb.model')
  if (file.exists('xgb.model')) file.remove('xgb.model')
  bst <- readRDS('xgb.model.rds')
  if (file.exists('xgb.model.rds')) file.remove('xgb.model.rds')
  nil_ptr <- new("externalptr")
  class(nil_ptr) <- "xgb.Booster.handle"
  expect_true(identical(bst$handle, nil_ptr))
  bst <- xgb.Booster.complete(bst)
  expect_true(!identical(bst$handle, nil_ptr))
  expect_equal(predict(bst.Tree, dtrain), predict(bst, dtrain), tolerance = float_tolerance)
})

test_that("xgb.model.dt.tree works with and without feature names", {
  .skip_if_vcd_not_available()
  names.dt.trees <- c("Tree", "Node", "ID", "Feature", "Split", "Yes", "No", "Missing", "Quality", "Cover")
  dt.tree <- xgb.model.dt.tree(feature_names = feature.names, model = bst.Tree)
  expect_equal(names.dt.trees, names(dt.tree))
  if (!flag_32bit)
    expect_equal(dim(dt.tree), c(188, 10))
  expect_output(str(dt.tree), 'Feature.*\\"Age\\"')

  dt.tree.0 <- xgb.model.dt.tree(model = bst.Tree)
  expect_equal(dt.tree, dt.tree.0)

  # when model contains no feature names:
  bst.Tree.x <- bst.Tree
  bst.Tree.x$feature_names <- NULL
  dt.tree.x <- xgb.model.dt.tree(model = bst.Tree.x)
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
  importance.Tree <- xgb.importance(feature_names = feature.names, model = bst.Tree)
  if (!flag_32bit)
    expect_equal(dim(importance.Tree), c(7, 4))
  expect_equal(colnames(importance.Tree), c("Feature", "Gain", "Cover", "Frequency"))
  expect_output(str(importance.Tree), 'Feature.*\\"Age\\"')

  importance.Tree.0 <- xgb.importance(model = bst.Tree)
  expect_equal(importance.Tree, importance.Tree.0, tolerance = float_tolerance)

  # when model contains no feature names:
  bst.Tree.x <- bst.Tree
  bst.Tree.x$feature_names <- NULL
  importance.Tree.x <- xgb.importance(model = bst.Tree)
  expect_equal(importance.Tree[, -1, with = FALSE], importance.Tree.x[, -1, with = FALSE],
               tolerance = float_tolerance)

  imp2plot <- xgb.plot.importance(importance_matrix = importance.Tree)
  expect_equal(colnames(imp2plot), c("Feature", "Gain", "Cover", "Frequency", "Importance"))
  xgb.ggplot.importance(importance_matrix = importance.Tree)

  # for multiclass
  imp.Tree <- xgb.importance(model = mbst.Tree)
  expect_equal(dim(imp.Tree), c(4, 4))

  trees <- seq(from = 0, by = 2, length.out = 2)
  importance <- xgb.importance(feature_names = feature.names, model = bst.Tree, trees = trees)

  importance_from_dump <- function() {
    model_text_dump <- xgb.dump(model = bst.Tree, with_stats = TRUE, trees = trees)
    imp <- xgb.model.dt.tree(
      feature_names = feature.names,
      text = model_text_dump,
      trees = trees
    )[
      Feature != "Leaf", .(
        Gain = sum(Quality),
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
  m <- xgboost::xgboost(
    data = as.matrix(data.frame(x = c(0, 1))),
    label = c(1, 2),
    nrounds = 1,
    base_score = 0.5
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

  # for multiclass
  imp.GLM <- xgb.importance(model = mbst.GLM)
  expect_equal(dim(imp.GLM), c(12, 3))
  expect_equal(imp.GLM$Class, rep(0:2, each = 4))
})

test_that("xgb.model.dt.tree and xgb.importance work with a single split model", {
  .skip_if_vcd_not_available()
  bst1 <- xgboost(data = sparse_matrix, label = label, max_depth = 1,
                  eta = 1, nthread = 2, nrounds = 1, verbose = 0,
                  objective = "binary:logistic")
  expect_error(dt <- xgb.model.dt.tree(model = bst1), regexp = NA) # no error
  expect_equal(nrow(dt), 3)
  expect_error(imp <- xgb.importance(model = bst1), regexp = NA) # no error
  expect_equal(nrow(imp), 1)
  expect_equal(imp$Gain, 1)
})

test_that("xgb.plot.tree works with and without feature names", {
  .skip_if_vcd_not_available()
  expect_silent(xgb.plot.tree(feature_names = feature.names, model = bst.Tree))
  expect_silent(xgb.plot.tree(model = bst.Tree))
})

test_that("xgb.plot.multi.trees works with and without feature names", {
  .skip_if_vcd_not_available()
  xgb.plot.multi.trees(model = bst.Tree, feature_names = feature.names, features_keep = 3)
  xgb.plot.multi.trees(model = bst.Tree, features_keep = 3)
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

test_that("check.deprecation works", {
  ttt <- function(a = NNULL, DUMMY = NULL, ...) {
    check.deprecation(...)
    as.list((environment()))
  }
  res <- ttt(a = 1, DUMMY = 2, z = 3)
  expect_equal(res, list(a = 1, DUMMY = 2))
  expect_warning(
    res <- ttt(a = 1, dummy = 22, z = 3)
  , "\'dummy\' is deprecated")
  expect_equal(res, list(a = 1, DUMMY = 22))
  expect_warning(
    res <- ttt(a = 1, dumm = 22, z = 3)
  , "\'dumm\' was partially matched to \'dummy\'")
  expect_equal(res, list(a = 1, DUMMY = 22))
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
