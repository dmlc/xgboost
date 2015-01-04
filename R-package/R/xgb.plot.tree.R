require(DiagrammeR)
require(stringr)
require(data.table)
require(magrittr)
text <- readLines('xgb.model.dump') %>% str_trim(side = "both")
position <- str_match(text, "booster") %>% is.na %>% not %>% which %>% c(length(text)+1)

extract <- function(x, pattern)  str_extract(x, pattern) %>% str_split("=") %>% lapply(function(x) x[2] %>% as.numeric) %>% unlist

#for(i in 1:(length(position)-1)){
i=1
  cat(paste("\n",i,"\n"))
  tree <- text[(position[i]+1):(position[i+1]-1)]
  paste(tree, collapse = "\n") %>% cat
notLeaf <- str_match(tree, "leaf") %>% is.na
leaf <- notLeaf %>% not %>% tree[.]
branch <- notLeaf %>% tree[.]
idBranch <- str_extract(branch, "\\d*:") %>% str_replace(":", "") %>% as.numeric
idLeaf <- str_extract(leaf, "\\d*:") %>% str_replace(":", "") %>% as.numeric 
featureBranch <- str_extract(branch, "f\\d*<") %>% str_replace("<", "") #%>% as.numeric 
featureLeaf <- rep("Leaf", length(leaf))
yesBranch <- extract(branch, "yes=\\d*")
yesLeaf <- rep(NA, length(leaf))
noBranch <- extract(branch, "no=\\d*")
noLeaf <- rep(NA, length(leaf))
missingBranch <- extract(branch, "missing=\\d+")
missingLeaf <- rep(NA, length(leaf))
qualityBranch <- extract(branch, "gain=\\d*\\.*\\d*")
qualityLeaf <- extract(leaf, "leaf=\\-*\\d*\\.*\\d*")
coverBranch <- extract(branch, "cover=\\d*\\.*\\d*")
coverLeaf <- extract(leaf, "cover=\\d*\\.*\\d*")
dt <- data.table(ID = c(idBranch, idLeaf), Feature = c(featureBranch, featureLeaf), Yes = c(yesBranch, yesLeaf), No = c(noBranch, noLeaf), Missing = c(missingBranch, missingLeaf), Quality = c(qualityBranch, qualityLeaf), Cover = c(coverBranch, coverLeaf))[order(ID)][,Tree:=i]

set(dt, j = "YesFeature", value = ifelse(is.na(dt[,Yes]),NA,dt[ID == dt[,Yes], ID]))
set(dt, j = "NoFeature", value = ifelse(is.na(dt[,No]),NA,dt[ID == dt[,No], ID]))
dtBranch <- dt[Feature!="Leaf"]

yesPath <- paste(dtBranch[,ID], "-->", dtBranch[,Yes], sep = "") 
noPath <- paste(dtBranch[,ID], "-->", dtBranch[,No], sep = "") 
missingPath <- paste(dtBranch[,ID], "-->|Missing|", dtBranch[,Missing], sep = "") 
yesPathStyle <- paste("style ", dtBranch[,Yes], " fill:#A2EB86, stroke:#04C4AB, stroke-width:2px", sep = "") 
noPathStyle <- paste("style ", dtBranch[,No], "  fill:#FFA070, stroke:#FF5E5E, stroke-width:2px", sep = "") 

path <- c(yesPath, noPath, yesPathStyle, noPathStyle) %>% .[order(.)] %>% paste(sep = "", collapse = ";") %>% paste("graph LR", .,collapse = "",sep = ";")

DiagrammeR(path, height = 400)
#}
