context('Test helper functions')

require(xgboost)
require(data.table)
require(Matrix)
require(vcd)

set.seed(1982)
data(Arthritis)
df <- data.table(Arthritis, keep.rownames = F)
df[,AgeDiscret := as.factor(round(Age / 10,0))]
df[,AgeCat := as.factor(ifelse(Age > 30, "Old", "Young"))]
df[,ID := NULL]
sparse_matrix <- sparse.model.matrix(Improved~.-1, data = df)
label <- df[, ifelse(Improved == "Marked", 1, 0)]

bst.Tree <- xgboost(data = sparse_matrix, label = label, max.depth = 9,
               eta = 1, nthread = 2, nround = 10, objective = "binary:logistic", booster = "gbtree")

bst.GLM <- xgboost(data = sparse_matrix, label = label,
                   eta = 1, nthread = 2, nround = 10, objective = "binary:logistic", booster = "gblinear")

feature.names <- colnames(sparse_matrix)

test_that("xgb.dump works", {
  expect_length(xgb.dump(bst.Tree), 172)
  expect_length(xgb.dump(bst.GLM), 14)
  expect_true(xgb.dump(bst.Tree, 'xgb.model.dump', with.stats = T))
  expect_true(file.exists('xgb.model.dump'))
  expect_gt(file.size('xgb.model.dump'), 8000)
})

test_that("xgb-attribute functionality", {
  val <- "my attribute value"
  list.val <- list(my_attr=val, a=123, b='ok')
  list.ch <- list.val[order(names(list.val))]
  list.ch <- lapply(list.ch, as.character)
  # proper input:
  expect_error(xgb.attr(bst.Tree, NULL))
  expect_error(xgb.attr(val, val))
  # set & get:
  expect_null(xgb.attr(bst.Tree, "asdf"))
  expect_null(xgb.attributes(bst.Tree)) # initially, expect no attributes
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
  expect_equal(xgb.attributes(bst), list.ch[c("a", "b")])
  xgb.attributes(bst) <- list(a=NULL, b=NULL)
  expect_null(xgb.attributes(bst))
})

test_that("xgb.model.dt.tree works with and without feature names", {
  names.dt.trees <- c("Tree", "Node", "ID", "Feature", "Split", "Yes", "No", "Missing", "Quality", "Cover")
  dt.tree <- xgb.model.dt.tree(feature_names = feature.names, model = bst.Tree)
  expect_equal(names.dt.trees, names(dt.tree))
  expect_equal(dim(dt.tree), c(162, 10))
  expect_output(str(xgb.model.dt.tree(model = bst.Tree)), 'Feature.*\\"3\\"')
})

test_that("xgb.importance works with and without feature names", {
  importance.Tree <- xgb.importance(feature_names = feature.names, model = bst.Tree)
  expect_equal(dim(importance.Tree), c(7, 4))
  expect_equal(colnames(importance.Tree), c("Feature", "Gain", "Cover", "Frequency"))
  expect_output(str(xgb.importance(model = bst.Tree)), 'Feature.*\\"3\\"')
  xgb.plot.importance(importance_matrix = importance.Tree)
})

test_that("xgb.importance works with GLM model", {
  importance.GLM <- xgb.importance(feature_names = feature.names, model = bst.GLM)
  expect_equal(dim(importance.GLM), c(10, 2))
  expect_equal(colnames(importance.GLM), c("Feature", "Weight"))
  xgb.importance(model = bst.GLM)
  xgb.plot.importance(importance.GLM)
})

test_that("xgb.plot.tree works with and without feature names", {
  xgb.plot.tree(feature_names = feature.names, model = bst.Tree)
  xgb.plot.tree(model = bst.Tree)
})

test_that("xgb.plot.multi.trees works with and without feature names", {
  xgb.plot.multi.trees(model = bst.Tree, feature_names = feature.names, features.keep = 3)
  xgb.plot.multi.trees(model = bst.Tree, features.keep = 3)
})

test_that("xgb.plot.deepness works", {
  xgb.plot.deepness(model = bst.Tree)
})
