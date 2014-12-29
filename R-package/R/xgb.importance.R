#' Show importance of features in a model
#' 
#' Read a xgboost model in text file format. 
#' Can be tree or linear model (text dump of linear model are only supported in dev version of Xgboost for now).
#' 
#' Return a data.table of the features with their weight.
#' #' 
#' @importFrom data.table data.table
#' @importFrom magrittr %>%
#' @importFrom data.table :=
#' @importFrom stringr str_extract
#' @param feature_names names of each feature as a character vector. Can be extracted from a sparse matrix (see example). If model dump already contains feature names, this argument should be \code{NULL}.
#' @param filename_dump the path to the text file storing the model.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#' 
#' #Both dataset are list with two items, a sparse matrix and labels (labels = outcome column which will be learned). 
#' #Each column of the sparse Matrix is a feature in one hot encoding format.
#' train <- agaricus.train
#' test <- agaricus.test
#' 
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nround = 2,objective = "binary:logistic")
#' xgb.dump(bst, 'xgb.model.dump', with.stats = T)
#' 
#' #agaricus.test$data@@Dimnames[[2]] represents the column names of the sparse matrix.
#' xgb.importance(agaricus.test$data@@Dimnames[[2]], 'xgb.model.dump')
#' 
#' @export
xgb.importance <- function(feature_names = NULL, filename_dump = NULL){  
  if (!class(feature_names) %in% c("character", "NULL")) {	   
    stop("feature_names: Has to be a vector of character or NULL if model dump already contain feature name. See help to see where to get it.")
  }
  if (class(filename_dump) != "character" & file.exists(filename_dump)) {
    stop("filename_dump: Has to be a path to the model dump file.")
  }
  text <- readLines(filename_dump)
  if(text[2] == "bias:"){
    result <- linearDump(feature_names, text)
  }  else {
    result <- treeDump(feature_names, text)
  }
  result
}

treeDump <- function(feature_names, text){   
  featureVec <- c()
  gainVec <- c()
  for(line in text){
    p <- str_extract(line, "\\[f.*<")
    if (!is.na(p)) {
      featureVec <- substr(p, 3, nchar(p)-1) %>% c(featureVec)
      gainVec <- str_extract(line, "gain.*,") %>%  substr(x = ., 6, nchar(.)-1) %>% as.numeric %>% c(gainVec)
    }
  }
  if(!is.null(feature_names)) {
    featureVec %<>% as.numeric %>% {c =.+1; feature_names[c]} #+1 because in R indexing start with 1 instead of 0.
  }
  #1. Reduce, 2. %, 3. reorder - bigger top, 4. remove temp col
  data.table(Feature = featureVec, Weight = gainVec)[,sum(Weight), by = Feature][, Weight:= V1 /sum(V1)][order(-rank(Weight))][,-2,with=F]
}

linearDump <- function(feature_names, text){
  which(text == "weight:") %>% {a=.+1;text[a:length(text)]} %>% as.numeric %>% data.table(Feature = feature_names, Weight = .)
}