#' Show importance of features in a model
#' 
#' Read a xgboost model text dump. 
#' Can be tree or linear model (text dump of linear model are only supported in dev version of \code{Xgboost} for now).
#' 
#' @importFrom data.table data.table
#' @importFrom magrittr %>%
#' @importFrom data.table :=
#' @importFrom stringr str_extract
#' @param feature_names names of each feature as a character vector. Can be extracted from a sparse matrix (see example). If model dump already contains feature names, this argument should be \code{NULL}.
#' @param filename_dump the path to the text file storing the model. Model dump must include the gain per feature and per tree (\code{with.stats = T} in function \code{xgb.dump}).
#'
#' @return A \code{data.table} of the features used in the model with their average gain (and their weight for boosted tree model) in the model.
#'
#' @details 
#' This is the function to understand the model trained (and through your model, your data).
#' 
#' Results are returned for both linear and tree models.
#' 
#' \code{data.table} is returned by the function. 
#' There are 3 columns :
#' \itemize{
#'   \item \code{Features} name of the features as provided in \code{feature_names} or already present in the model dump.
#'   \item \code{Gain} contribution of each feature to the model. For boosted tree model, each gain of each feature of each tree is taken into account, then average per feature to give a vision of the entire model. Highest percentage means most important feature regarding the \code{label} used for the training.
#'   \item \code{Weight} percentage representing the relative number of times a feature have been taken into trees. \code{Gain} should be prefered to search the most important feature. For boosted linear model, this column has no meaning.
#' }
#' 
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
    stop("feature_names: Has to be a vector of character or NULL if the model dump already contains feature name. Look at this function documentation to see where to get feature names.")
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
  data.table(Feature = featureVec, Weight = gainVec)[,list(sum(Weight), .N), by = Feature][, Gain:= V1/sum(V1)][,Weight:= N/sum(N)][order(-rank(Gain))][,-c(2,3), with = F]
}

linearDump <- function(feature_names, text){
  which(text == "weight:") %>% {a=.+1;text[a:length(text)]} %>% as.numeric %>% data.table(Feature = feature_names, Weight = .)
}