library(stringr)
library(data.table)



data(agaricus.train, package='xgboost')

#Both dataset are list with two items, a sparse matrix and labels
#(labels = outcome column which will be learned).
#Each column of the sparse Matrix is a feature in one hot encoding format.
train <- agaricus.train

bst <- xgboost(data = train$data, label = train$label, max.depth = 5,
               eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")

#agaricus.test$data@Dimnames[[2]] represents the column names of the sparse matrix.
tree.matrix <- xgb.model.dt.tree(agaricus.train$data@Dimnames[[2]], model = bst)


# first number of the path represents the tree, then the following numbers are related to the path to follow

# root init
root.nodes <- tree.matrix[str_detect(ID, "\\d+-0"), ID]
tree.matrix[ID == root.nodes, Abs.Position:=root.nodes]

precedent.nodes <- root.nodes

while(tree.matrix[,sum(is.na(Abs.Position))] > 0) {
  yes.row.nodes <- tree.matrix[Abs.Position %in% precedent.nodes & !is.na(Yes)]
  no.row.nodes <- tree.matrix[Abs.Position %in% precedent.nodes & !is.na(No)]
  yes.nodes.abs.pos <- yes.row.nodes[, Abs.Position] %>% paste0("-0")
  no.nodes.abs.pos <- no.row.nodes[, Abs.Position] %>% paste0("-1")
  
  tree.matrix[ID == yes.row.nodes[, Yes], Abs.Position := yes.nodes.abs.pos]
  tree.matrix[ID == no.row.nodes[, No], Abs.Position := no.nodes.abs.pos]
  precedent.nodes <- c(yes.nodes.abs.pos, no.nodes.abs.pos)
}

tree.matrix



