context('Test helper functions')

require(xgboost)
require(data.table)
require(Matrix)
require(vcd)

data(Arthritis)
data(agaricus.train, package='xgboost')
df <- data.table(Arthritis, keep.rownames = F)
df[,AgeDiscret:= as.factor(round(Age/10,0))]
df[,AgeCat:= as.factor(ifelse(Age > 30, "Old", "Young"))]
df[,ID:=NULL]
sparse_matrix = sparse.model.matrix(Improved~.-1, data = df)
output_vector = df[,Y:=0][Improved == "Marked",Y:=1][,Y]
bst <- xgboost(data = sparse_matrix, label = output_vector, max.depth = 9,
               eta = 1, nthread = 2, nround = 10,objective = "binary:logistic")


test_that("xgb.dump works", {
  capture.output(print(xgb.dump(bst)))
})

test_that("xgb.importance works", {
  xgb.dump(bst, 'xgb.model.dump', with.stats = T)
  importance <- xgb.importance(sparse_matrix@Dimnames[[2]], 'xgb.model.dump')
  expect_equal(dim(importance), c(7, 4))
})

test_that("xgb.plot.tree works", {
  xgb.plot.tree(agaricus.train$data@Dimnames[[2]], model = bst)
})