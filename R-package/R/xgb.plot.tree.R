require(DiagrammeR)
require(stringr)
require(data.table)
require(magrittr)
text <- readLines('xgb.model.dump') %>% str_trim(side = "both")
position <- str_match(text, "booster") %>% is.na %>% not %>% which %>% c(length(text)+1)

extract <- function(x, pattern)  str_extract(x, pattern) %>% str_split("=") %>% lapply(function(x) x[2] %>% as.numeric) %>% unlist

addTreeId <- function(x, i) paste(i,x,sep = "-")

allTrees <- data.table()

for(i in 1:(length(position)-1)){
  
  tree <- text[(position[i]+1):(position[i+1]-1)]
  
  notLeaf <- str_match(tree, "leaf") %>% is.na
  leaf <- notLeaf %>% not %>% tree[.]
  branch <- notLeaf %>% tree[.]
  idBranch <- str_extract(branch, "\\d*:") %>% str_replace(":", "") %>% addTreeId(i)
  idLeaf <- str_extract(leaf, "\\d*:") %>% str_replace(":", "") %>% addTreeId(i)
  featureBranch <- str_extract(branch, "f\\d*<") %>% str_replace("<", "") %>% str_replace("f", "") %>% as.numeric 
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
  
  dt[Feature!="Leaf" ,noPath:= paste(ID,"[", Feature, "]-->|> ", Split, "|", No, "[", NoFeature, "]", sep = "")]
  
  #missingPath <- paste(dtBranch[,ID], "-->|Missing|", dtBranch[,Missing], sep = "") 
  
  allTrees <- rbindlist(list(allTrees, dt), use.names = T, fill = F)
}

styles <- "classDef greenNode fill:#A2EB86, stroke:#04C4AB, stroke-width:2px;classDef redNode fill:#FFA070, stroke:#FF5E5E, stroke-width:2px;"

yes <- allTrees[Feature!="Leaf", c(Yes)] %>% paste(collapse = ",") %>% paste("class ", ., " greenNode;", sep = "")

no <- allTrees[Feature!="Leaf", c(No)] %>% paste(collapse = ",") %>% paste("class ", ., " redNode;", sep = "")

path <- allTrees[Feature!="Leaf", c(yesPath, noPath)] %>% .[order(.)] %>% paste(sep = "", collapse = ";") %>% paste("graph LR", .,collapse = "", sep = ";") %>% paste(";", styles, yes, no, collapse = ";", sep = "")

DiagrammeR(path, height =700)
#}
