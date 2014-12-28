#' Show importance of features in a model
#' 
#' Read a xgboost model in text file format. Return a data.table of the features with their weight.
#' 
#' @importFrom data.table data.table
#' @importFrom magrittr %>%
#' @importFrom data.table :=
#' @param feature_names names of each feature as a character vector. Can be extracted from a sparse matrix.
#' @param filename_dump the name of the text file.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' 
#' #Both dataset are list with two items, a sparse matrix and labels (outcome column which will be learned). 
#' #Each column of the sparse Matrix is a feature in one hot encoding format.
#' train <- agaricus.train
#' test <- agaricus.test
#' 
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nround = 2,objective = "binary:logistic")
#' xgb.dump(bst, 'xgb.model.dump')
#' 
#' #agaricus.test$data@@Dimnames[[2]] represents the column name of the sparse matrix.
#' xgb.importance(agaricus.test$data@@Dimnames[[2]], 'xgb.model.dump')
#' 
#' @export
xgb.importance <- function(feature_names, filename_dump){  
  text <- readLines(filename_dump)  
  if(text[2] == "bias:"){
    result <- linearDump(feature_names, text)
  }  else {
    result <- treeDump(feature_names, text)
  }
  result
}

treeDump <- function(feature_names, text){  
  result <- c()
  for(line in text){
    p <- regexec("\\[f.*\\]", line) %>% regmatches(line, .)
    if (length(p[[1]]) > 0) {      
      splits <- sub("\\[f", "", p[[1]]) %>% sub("\\]", "", .) %>% strsplit("<") %>% .[[1]] %>% as.numeric
      result <- c(result, feature_names[splits[1]+ 1])
    }
  }
  #1. Reduce, 2. %, 3. reorder - bigger top, 4. remove temp col
  data.table(Feature = result)[,.N, by = Feature][, Weight:= N /sum(N)][order(-rank(Weight))][,-2,with=F]
}

linearDump <- function(feature_names, text){
  which(text == "weight:") %>% {a=.+1;text[a:length(text)]} %>% as.numeric %>% data.table(Feature = feature_names, Weight = .)
}