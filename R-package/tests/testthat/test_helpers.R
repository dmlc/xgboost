context('Test helper functions')

require(xgboost)
require(data.table)
require(Matrix)
require(vcd)

set.seed(1982)
data(Arthritis)
data(agaricus.train, package='xgboost')
df <- data.table(Arthritis, keep.rownames = F)
df[,AgeDiscret := as.factor(round(Age / 10,0))]
df[,AgeCat := as.factor(ifelse(Age > 30, "Old", "Young"))]
df[,ID := NULL]
sparse_matrix <- sparse.model.matrix(Improved~.-1, data = df)
output_vector <- df[,Y := 0][Improved == "Marked",Y := 1][,Y]
bst <- xgboost(data = sparse_matrix, label = output_vector, max.depth = 9,
               eta = 1, nthread = 2, nround = 10, objective = "binary:logistic")

feature.names <- agaricus.train$data@Dimnames[[2]]

test_that("xgb.dump works", {
  capture.output(print(xgb.dump(bst)))
  expect_true(xgb.dump(bst, 'xgb.model.dump', with.stats = T))
})

test_that("xgb.model.dt.tree works with and without feature names", {
  names.dt.trees <- c("ID", "Feature", "Split", "Yes", "No", "Missing", "Quality", "Cover",
   "Tree", "Yes.Feature", "Yes.Cover", "Yes.Quality", "No.Feature", "No.Cover", "No.Quality")
  dt.tree <- xgb.model.dt.tree(feature_names = feature.names, model = bst)
  expect_equal(names.dt.trees, names(dt.tree))
  expect_equal(dim(dt.tree), c(162, 15))
  xgb.model.dt.tree(model = bst)
})

test_that("xgb.importance works with and without feature names", {
  importance <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst)
  expect_equal(dim(importance), c(7, 4))
  expect_equal(colnames(importance), c("Feature", "Gain", "Cover", "Frequency"))
  xgb.importance(model = bst)
})

test_that("xgb.importance works with GLM model", {
  bst.GLM <- xgboost(data = sparse_matrix, label = output_vector,
                 eta = 1, nthread = 2, nround = 10, objective = "binary:logistic", booster = "gblinear")
  importance.GLM <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst.GLM)
  expect_equal(dim(importance.GLM), c(10, 2))
  expect_equal(colnames(importance.GLM), c("Feature", "Weight"))
  xgb.importance(model = bst.GLM)
})

test_that("xgb.plot.tree works with and without feature names", {
  xgb.plot.tree(feature_names = feature.names, model = bst)
  xgb.plot.tree(model = bst)
})

test_that("xgb.plot.multi.trees works with and without feature names", {
  xgb.plot.multi.trees(model = bst, feature_names = feature.names, features.keep = 3)
  xgb.plot.multi.trees(model = bst, features.keep = 3)
})
test_that("xgb.plot.deepness works", {
  xgb.plot.deepness(model = bst)
})
