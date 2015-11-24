library(stringr)
library(data.table)
library(xgboost)


data(agaricus.train, package='xgboost')

#Both dataset are list with two items, a sparse matrix and labels
#(labels = outcome column which will be learned).
#Each column of the sparse Matrix is a feature in one hot encoding format.
train <- agaricus.train

bst <- xgboost(data = train$data, label = train$label, max.depth = 2,
               eta = 1, nthread = 2, nround = 4, objective = "binary:logistic")

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
  yes.nodes.abs.pos <- yes.row.nodes[, Abs.Position] %>% paste0("_0")
  no.nodes.abs.pos <- no.row.nodes[, Abs.Position] %>% paste0("_1")
  
  tree.matrix[ID == yes.row.nodes[, Yes], Abs.Position := yes.nodes.abs.pos]
  tree.matrix[ID == no.row.nodes[, No], Abs.Position := no.nodes.abs.pos]
  precedent.nodes <- c(yes.nodes.abs.pos, no.nodes.abs.pos)
}

tree.matrix[!is.na(Yes),Yes:= paste0(Abs.Position, "_0")]
tree.matrix[!is.na(No),No:= paste0(Abs.Position, "_1")]
tree.matrix[,ID:= Abs.Position]

tree.matrix[,Abs.Position:=substr(Abs.Position, nchar(Tree)+2, nchar(Abs.Position))]
keepN <- 3
tree.matrix <- tree.matrix[,sum(Quality),by = .(Abs.Position, Feature)][order(-V1)][,.(paste0(Feature[1:min(length(Feature), keepN)], " (", V1[1:min(length(V1), keepN)], ")") %>% paste0(collapse = "\n")), by=Abs.Position]

tree.matrix[Feature!="Leaf" ,yesPath:= paste(ID,"(", Feature, "<br/>Cover: ", Cover, "<br/>Gain: ", Quality, ")-->|< ", Split, "|", Yes, ">", Yes.Feature, "]", sep = "")]

tree.matrix[Feature!="Leaf" ,noPath:= paste(ID,"(", Feature, ")-->|>= ", Split, "|", No, ">", No.Feature, "]", sep = "")]

tree.matrix[, Yes:= Abs.Position %>% paste0("_0")][, No:= Abs.Position %>% paste0("_1")]

CSSstyle <- "classDef greenNode fill:#A2EB86, stroke:#04C4AB, stroke-width:2px\nclassDef redNode fill:#FFA070, stroke:#FF5E5E, stroke-width:2px"  


yes <- tree.matrix[Feature!="Leaf", c(Yes)] %>% paste(collapse = ",") %>% paste("class ", ., " greenNode", sep = "")

no <- tree.matrix[Feature!="Leaf", c(No)] %>% paste(collapse = ",") %>% paste("class ", ., " redNode", sep = "")

path <- tree.matrix[Feature!="Leaf", c(yesPath, noPath)] %>% .[order(.)] %>% paste(sep = "", collapse = "\n") %>% paste("graph LR", .,collapse = "", sep = "\n") %>% paste(CSSstyle, yes, no, sep = "\n")
DiagrammeR::mermaid(path)

# path <- "graph LR;0-0-0(spore-print-color=green)-->|>= 2.00001|0-0-0-1>Leaf"
# setnames(tree.matrix, old = c("ID", "Yes", "No"), c("nodes", "edge_from", "edge_to"))
