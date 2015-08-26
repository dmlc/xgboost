#' Plot feature importance bar graph
#' 
#' Read a data.table containing feature importance details and plot it.
#' 
#' @importFrom magrittr %>%
#' @param importance_matrix a \code{data.table} returned by the \code{xgb.importance} function.
#' @param numberOfClusters a \code{numeric} vector containing the min and the max range of the possible number of clusters of bars.
#'
#' @return A \code{ggplot2} bar graph representing each feature by a horizontal bar. Longer is the bar, more important is the feature. Features are classified by importance and clustered by importance. The group is represented through the color of the bar.
#'
#' @details 
#' The purpose of this function is to easily represent the importance of each feature of a model.
#' The function return a ggplot graph, therefore each of its characteristic can be overriden (to customize it).
#' In particular you may want to override the title of the graph. To do so, add \code{+ ggtitle("A GRAPH NAME")} next to the value returned by this function. 
#'   
#' @examples
#' data(agaricus.train, package='xgboost')
#' 
#' #Both dataset are list with two items, a sparse matrix and labels 
#' #(labels = outcome column which will be learned). 
#' #Each column of the sparse Matrix is a feature in one hot encoding format.
#' train <- agaricus.train
#' 
#' bst <- xgboost(data = train$data, label = train$label, max.depth = 2, 
#'                eta = 1, nthread = 2, nround = 2,objective = "binary:logistic")
#' 
#' #train$data@@Dimnames[[2]] represents the column names of the sparse matrix.
#' importance_matrix <- xgb.importance(train$data@@Dimnames[[2]], model = bst)
#' xgb.plot.importance(importance_matrix)
#' 
#' @export
xgb.plot.importance <- function(importance_matrix = NULL, numberOfClusters = c(1:10)){
  if (!"data.table" %in% class(importance_matrix))  {     
    stop("importance_matrix: Should be a data.table.")
  }
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required for plotting the importance", call. = FALSE)
  }
  if (!requireNamespace("Ckmeans.1d.dp", quietly = TRUE)) {
    stop("Ckmeans.1d.dp package is required for plotting the importance", call. = FALSE)
  }

  # To avoid issues in clustering when co-occurences are used
  importance_matrix <- importance_matrix[, .(Gain = sum(Gain)), by = Feature]
  
  clusters <- suppressWarnings(Ckmeans.1d.dp::Ckmeans.1d.dp(importance_matrix[,Gain], numberOfClusters))
  importance_matrix[,"Cluster":=clusters$cluster %>% as.character]
    
  plot <- ggplot2::ggplot(importance_matrix, ggplot2::aes(x=stats::reorder(Feature, Gain), y = Gain, width= 0.05), environment = environment())+  ggplot2::geom_bar(ggplot2::aes(fill=Cluster), stat="identity", position="identity") + ggplot2::coord_flip() + ggplot2::xlab("Features") + ggplot2::ylab("Gain") + ggplot2::ggtitle("Feature importance") + ggplot2::theme(plot.title = ggplot2::element_text(lineheight=.9, face="bold"), panel.grid.major.y = ggplot2::element_blank() )
  
  return(plot)  
}

# Avoid error messages during CRAN check.
# The reason is that these variables are never declared
# They are mainly column names inferred by Data.table...
globalVariables(c("Feature", "Gain", "Cluster", "ggplot", "aes", "geom_bar", "coord_flip", "xlab", "ylab", "ggtitle", "theme", "element_blank", "element_text"))
