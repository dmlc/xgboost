#' Plot a boosted tree model
#' 
#' Read a xgboost model text dump. 
#' Only works for boosted tree model (not linear model).
#' 
#' @importFrom data.table data.table
#' @importFrom data.table set
#' @importFrom data.table rbindlist
#' @importFrom magrittr %>%
#' @importFrom magrittr not
#' @importFrom magrittr add
#' @importFrom data.table :=
#' @importFrom stringr str_extract
#' @importFrom stringr str_split
#' @importFrom stringr str_extract
#' @importFrom stringr str_trim
#' @importFrom DiagrammeR DiagrammeR
#' @param feature_names names of each feature as a character vector. Can be extracted from a sparse matrix (see example). If model dump already contains feature names, this argument should be \code{NULL}.
#' @param filename_dump the path to the text file storing the model. Model dump must include the gain per feature and per tree (\code{with.stats = T} in function \code{xgb.dump}).
#' @param n_first_tree limit the plot to the n first trees.
#'
#' @return A \code{data.table} of the features used in the model with their average gain (and their weight for boosted tree model) in the model.
#'
#' @details 
#' This is the function to plot the trees growned.
#' It uses Mermaid JS library for that purpose.
#' Performance can be low for huge models.
#' 
#' 
#' @examples
#' data(agaricus.train, package='xgboost')
#' 
#' #Both dataset are list with two items, a sparse matrix and labels (labels = outcome column which will be learned). 
#' #Each column of the sparse Matrix is a feature in one hot encoding format.
#' train <- agaricus.train
#' 
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nround = 2,objective = "binary:logistic")
#' xgb.dump(bst, 'xgb.model.dump', with.stats = T)
#' 
#' #agaricus.test$data@@Dimnames[[2]] represents the column names of the sparse matrix.
#' xgb.plot.tree(agaricus.train$data@@Dimnames[[2]], 'xgb.model.dump')
#' 
#' @export
xgb.plot.tree <- function(feature_names = NULL, filename_dump = NULL, n_first_tree = NULL){
  
  if (!class(feature_names) %in% c("character", "NULL")) {     
    stop("feature_names: Has to be a vector of character or NULL if the model dump already contains feature name. Look at this function documentation to see where to get feature names.")
  }
  if (class(filename_dump) != "character" || !file.exists(filename_dump)) {
    stop("filename_dump: Has to be a path to the model dump file.")
  }
  if (!class(n_first_tree) %in% c("numeric", "NULL") | length(n_first_tree) > 1) {
    stop("n_first_tree: Has to be a numeric vector of size 1.")
  }
  
  text <- readLines(filename_dump) %>% str_trim(side = "both")
  position <- str_match(text, "booster") %>% is.na %>% not %>% which %>% c(length(text)+1)
  
  extract <- function(x, pattern)  str_extract(x, pattern) %>% str_split("=") %>% lapply(function(x) x[2] %>% as.numeric) %>% unlist
  
  n_round <- min(length(position) - 1, n_first_tree)
  
  addTreeId <- function(x, i) paste(i,x,sep = "-")
  
  allTrees <- data.table()
  
  for(i in 1:n_round){
    
    tree <- text[(position[i]+1):(position[i+1]-1)]
    
    notLeaf <- str_match(tree, "leaf") %>% is.na
    leaf <- notLeaf %>% not %>% tree[.]
    branch <- notLeaf %>% tree[.]
    idBranch <- str_extract(branch, "\\d*:") %>% str_replace(":", "") %>% addTreeId(i)
    idLeaf <- str_extract(leaf, "\\d*:") %>% str_replace(":", "") %>% addTreeId(i)
    featureBranch <- str_extract(branch, "f\\d*<") %>% str_replace("<", "") %>% str_replace("f", "") %>% as.numeric 
    if(!is.null(feature_names)){
      featureBranch <- feature_names[featureBranch + 1]
    }
    featureLeaf <- rep("Leaf", length(leaf))
    splitBranch <- str_extract(branch, "<\\d*\\.*\\d*\\]") %>% str_replace("<", "") %>% str_replace("\\]", "") 
    splitLeaf <- rep(NA, length(leaf)) 
    yesBranch <- extract(branch, "yes=\\d*") %>% addTreeId(i)
    yesLeaf <- rep(NA, length(leaf)) 
    noBranch <- extract(branch, "no=\\d*") %>% addTreeId(i)
    noLeaf <- rep(NA, length(leaf))
    missingBranch <- extract(branch, "missing=\\d+") %>% addTreeId(i)
    missingLeaf <- rep(NA, length(leaf))
    qualityBranch <- extract(branch, "gain=\\d*\\.*\\d*")
    qualityLeaf <- extract(leaf, "leaf=\\-*\\d*\\.*\\d*")
    coverBranch <- extract(branch, "cover=\\d*\\.*\\d*")
    coverLeaf <- extract(leaf, "cover=\\d*\\.*\\d*")
    dt <- data.table(ID = c(idBranch, idLeaf), Feature = c(featureBranch, featureLeaf), Split = c(splitBranch, splitLeaf), Yes = c(yesBranch, yesLeaf), No = c(noBranch, noLeaf), Missing = c(missingBranch, missingLeaf), Quality = c(qualityBranch, qualityLeaf), Cover = c(coverBranch, coverLeaf))[order(ID)][,Tree:=i]
    
    set(dt, i = which(dt[,Feature]!= "Leaf"), j = "YesFeature", value = dt[ID == dt[,Yes], Feature])
    set(dt, i = which(dt[,Feature]!= "Leaf"), j = "NoFeature", value = dt[ID == dt[,No], Feature])
    
    dt[Feature!="Leaf" ,yesPath:= paste(ID,"[", Feature, "]-->|< ", Split, "|", Yes, "[", YesFeature, "]", sep = "")]
    
    dt[Feature!="Leaf" ,noPath:= paste(ID,"[", Feature, "]-->|>= ", Split, "|", No, "[", NoFeature, "]", sep = "")]
    
    #missingPath <- paste(dtBranch[,ID], "-->|Missing|", dtBranch[,Missing], sep = "") 
    
    allTrees <- rbindlist(list(allTrees, dt), use.names = T, fill = F)
  }
  
  styles <- "classDef greenNode fill:#A2EB86, stroke:#04C4AB, stroke-width:2px;classDef redNode fill:#FFA070, stroke:#FF5E5E, stroke-width:2px"
  
  yes <- allTrees[Feature!="Leaf", c(Yes)] %>% paste(collapse = ",") %>% paste("class ", ., " greenNode", sep = "")
  
  no <- allTrees[Feature!="Leaf", c(No)] %>% paste(collapse = ",") %>% paste("class ", ., " redNode", sep = "")
  
  path <- allTrees[Feature!="Leaf", c(yesPath, noPath)] %>% .[order(.)] %>% paste(sep = "", collapse = ";") %>% paste("graph LR", .,collapse = "", sep = ";") %>% paste(styles, yes, no, sep = ";")
  
  DiagrammeR(path)
}
