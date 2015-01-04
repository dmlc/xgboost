require(DiagrammeR)
require(stringr)
require(data.table)
require(magrittr)
text <- readLines('xgb.model.dump') %>% str_trim(side = "both")
position <- str_match(text, "booster") %>% is.na %>% not %>% which %>% c(length(text)+1)

extract <- function(x, pattern)  str_extract(x, pattern) %>% str_split("=") %>% lapply(function(x) x[2]) %>% unlist %>% as.numeric

#for(i in 1:(length(position)-1)){
i=1
  cat(paste("\n",i,"\n"))
  tree <- text[(position[i]+1):(position[i+1]-1)]
  paste(tree, collapse = "\n") %>% cat
branch <- str_match(tree, "leaf") %>% is.na %>% tree[.]
id <- str_extract(branch, "\\d*:") %>% str_replace(":", "") %>% as.numeric
feature <- str_extract(branch, "\\[.*\\]")
yes <- extract(branch, "yes=\\d*") 
no <- extract(branch, "no=\\d*")
missing <- extract(branch, "missing=\\d+")
gain <- extract(branch, "gain=\\d*\\.*\\d*")
cover <- extract(branch, "cover=\\d*\\.*\\d*")
dt <- data.table(ID = id, Feature = feature, Yes = yes, No = no, Missing = missing, Gain = gain, Cover = cover)
#}
