context('Test helper functions')

require(xgboost)
require(data.table)
require(Matrix)
require(vcd, quietly = TRUE)

set.seed(1982)
data(Arthritis)
df <- data.table(Arthritis, keep.rownames = F)
df[,AgeDiscret := as.factor(round(Age / 10,0))]
df[,AgeCat := as.factor(ifelse(Age > 30, "Old", "Young"))]
df[,ID := NULL]
sparse_matrix <- sparse.model.matrix(Improved~.-1, data = df)
label <- df[, ifelse(Improved == "Marked", 1, 0)]

bst.Tree <- xgboost(data = sparse_matrix, label = label, max_depth = 9,
                    eta = 1, nthread = 2, nrounds = 10, verbose = 0,
                    objective = "binary:logistic", booster = "gbtree")

bst.GLM <- xgboost(data = sparse_matrix, label = label,
                   eta = 1, nthread = 2, nrounds = 10, verbose = 0,
                   objective = "binary:logistic", booster = "gblinear")

feature.names <- colnames(sparse_matrix)

test_that("xgb.dump works", {
  expect_length(xgb.dump(bst.Tree), 172)
  expect_true(xgb.dump(bst.Tree, 'xgb.model.dump', with_stats = T))
  expect_true(file.exists('xgb.model.dump'))
  expect_gt(file.size('xgb.model.dump'), 8000)

  # JSON format
  dmp <- xgb.dump(bst.Tree, dump_format = "json")
  expect_length(dmp, 1)
  expect_length(grep('nodeid', strsplit(dmp, '\n')[[1]]), 162)
})

test_that("xgb.dump works for gblinear", {
  expect_length(xgb.dump(bst.GLM), 14)
  # also make sure that it works properly for a sparse model where some coefficients 
  # are 0 from setting large L1 regularization:
  bst.GLM.sp <- xgboost(data = sparse_matrix, label = label, eta = 1, nthread = 2, nrounds = 1, 
                        alpha=2, objective = "binary:logistic", booster = "gblinear")
  d.sp <- xgb.dump(bst.GLM.sp)
  expect_length(d.sp, 14)
  expect_gt(sum(d.sp == "0"), 0)

  # JSON format
  dmp <- xgb.dump(bst.GLM.sp, dump_format = "json")
  expect_length(dmp, 1)
  expect_length(grep('\\d', strsplit(dmp, '\n')[[1]]), 11)
})

test_that("xgb-attribute functionality", {
  val <- "my attribute value"
  list.val <- list(my_attr=val, a=123, b='ok')
  list.ch <- list.val[order(names(list.val))]
  list.ch <- lapply(list.ch, as.character)
  # note: iter is 0-index in xgb attributes
  list.default <- list(niter = "9")
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
  expect_equal(xgb.attr(bst, "my_attr"), val)
  expect_equal(xgb.attributes(bst), list.ch)
  # deletion:
  xgb.attr(bst, "my_attr") <- NULL
  expect_null(xgb.attr(bst, "my_attr"))
  expect_equal(xgb.attributes(bst), list.ch[c("a", "b", "niter")])
  xgb.attributes(bst) <- list(a=NULL, b=NULL)
  expect_equal(xgb.attributes(bst), list.default)
  xgb.attributes(bst) <- list(niter=NULL)
  expect_null(xgb.attributes(bst))
})

if (grepl('Windows', Sys.info()[['sysname']]) || grepl('Linux', Sys.info()[['sysname']]) || grepl('Darwin', Sys.info()[['sysname']])) {
    test_that("xgb-attribute numeric precision", {
      # check that lossless conversion works with 17 digits
      # numeric -> character -> numeric
      X <- 10^runif(100, -20, 20)
      X2X <- as.numeric(format(X, digits = 17))
      expect_identical(X, X2X)
      # retrieved attributes to be the same as written
      for (x in X) {
        xgb.attr(bst.Tree, "x") <- x
        expect_identical(as.numeric(xgb.attr(bst.Tree, "x")), x)
        xgb.attributes(bst.Tree) <- list(a = "A", b = x)
        expect_identical(as.numeric(xgb.attr(bst.Tree, "b")), x)
      }
    })
}

test_that("xgb.Booster serializing as R object works", {
  saveRDS(bst.Tree, 'xgb.model.rds')
  bst <- readRDS('xgb.model.rds')
  dtrain <- xgb.DMatrix(sparse_matrix, label = label)
  expect_equal(predict(bst.Tree, dtrain), predict(bst, dtrain))
  expect_equal(xgb.dump(bst.Tree), xgb.dump(bst))
  xgb.save(bst, 'xgb.model')
  nil_ptr <- new("externalptr")
  class(nil_ptr) <- "xgb.Booster.handle"
  expect_true(identical(bst$handle, nil_ptr))
  bst <- xgb.Booster.complete(bst)
  expect_true(!identical(bst$handle, nil_ptr))
  expect_equal(predict(bst.Tree, dtrain), predict(bst, dtrain))
})

test_that("xgb.model.dt.tree works with and without feature names", {
  names.dt.trees <- c("Tree", "Node", "ID", "Feature", "Split", "Yes", "No", "Missing", "Quality", "Cover")
  dt.tree <- xgb.model.dt.tree(feature_names = feature.names, model = bst.Tree)
  expect_equal(names.dt.trees, names(dt.tree))
  expect_equal(dim(dt.tree), c(162, 10))
  expect_output(str(dt.tree), 'Feature.*\\"Age\\"')
  
  dt.tree.0 <- xgb.model.dt.tree(model = bst.Tree)
  expect_equal(dt.tree, dt.tree.0)
  
  # when model contains no feature names:
  bst.Tree.x <- bst.Tree
  bst.Tree.x$feature_names <- NULL
  dt.tree.x <- xgb.model.dt.tree(model = bst.Tree.x)
  expect_output(str(dt.tree.x), 'Feature.*\\"3\\"')
  expect_equal(dt.tree[, -4, with=FALSE], dt.tree.x[, -4, with=FALSE])
})

test_that("xgb.model.dt.tree throws error for gblinear", {
  expect_error(xgb.model.dt.tree(model = bst.GLM))
})

test_that("xgb.importance works with and without feature names", {
  importance.Tree <- xgb.importance(feature_names = feature.names, model = bst.Tree)
  expect_equal(dim(importance.Tree), c(7, 4))
  expect_equal(colnames(importance.Tree), c("Feature", "Gain", "Cover", "Frequency"))
  expect_output(str(importance.Tree), 'Feature.*\\"Age\\"')
  
  importance.Tree.0 <- xgb.importance(model = bst.Tree)
  expect_equal(importance.Tree, importance.Tree.0)
  
  # when model contains no feature names:
  bst.Tree.x <- bst.Tree
  bst.Tree.x$feature_names <- NULL
  importance.Tree.x <- xgb.importance(model = bst.Tree)
  expect_equal(importance.Tree[, -1, with=FALSE], importance.Tree.x[, -1, with=FALSE])
  
  imp2plot <- xgb.plot.importance(importance_matrix = importance.Tree)
  expect_equal(colnames(imp2plot), c("Feature", "Gain", "Cover", "Frequency", "Importance"))
  xgb.ggplot.importance(importance_matrix = importance.Tree)
})

test_that("xgb.importance works with GLM model", {
  importance.GLM <- xgb.importance(feature_names = feature.names, model = bst.GLM)
  expect_equal(dim(importance.GLM), c(10, 2))
  expect_equal(colnames(importance.GLM), c("Feature", "Weight"))
  xgb.importance(model = bst.GLM)
  imp2plot <- xgb.plot.importance(importance.GLM)
  expect_equal(colnames(imp2plot), c("Feature", "Weight", "Importance"))
  xgb.ggplot.importance(importance.GLM)
})

test_that("xgb.plot.tree works with and without feature names", {
  xgb.plot.tree(feature_names = feature.names, model = bst.Tree)
  xgb.plot.tree(model = bst.Tree)
})

test_that("xgb.plot.multi.trees works with and without feature names", {
  xgb.plot.multi.trees(model = bst.Tree, feature_names = feature.names, features_keep = 3)
  xgb.plot.multi.trees(model = bst.Tree, features_keep = 3)
})

test_that("xgb.plot.deepness works", {
  d2p <- xgb.plot.deepness(model = bst.Tree)
  expect_equal(colnames(d2p), c("ID", "Tree", "Depth", "Cover", "Weight"))
  xgb.plot.deepness(model = bst.Tree, which = "med.depth")
  xgb.ggplot.deepness(model = bst.Tree)
})

test_that("check.deprecation works", {
  ttt <- function(a = NNULL, DUMMY=NULL, ...) {
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
