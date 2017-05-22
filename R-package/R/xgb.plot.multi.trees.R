#' Project all trees on one tree and plot it
#' 
#' Visualization of the ensemble of trees as a single collective unit.
#'
#' @param model produced by the \code{xgb.train} function.
#' @param feature_names names of each feature as a \code{character} vector.
#' @param features_keep number of features to keep in each position of the multi trees.
#' @param plot_width width in pixels of the graph to produce
#' @param plot_height height in pixels of the graph to produce
#' @param ... currently not used
#' 
#' @return Two graphs showing the distribution of the model deepness.
#' 
#' @details
#' 
#' This function tries to capture the complexity of a gradient boosted tree model 
#' in a cohesive way by compressing an ensemble of trees into a single tree-graph representation.
#' The goal is to improve the interpretability of a model generally seen as black box.
#' 
#' Note: this function is applicable to tree booster-based models only.
#' 
#' It takes advantage of the fact that the shape of a binary tree is only defined by 
#' its depth (therefore, in a boosting model, all trees have similar shape). 
#' 
#' Moreover, the trees tend to reuse the same features.
#' 
#' The function projects each tree onto one, and keeps for each position the 
#' \code{features_keep} first features (based on the Gain per feature measure).
#' 
#' This function is inspired by this blog post:
#' \url{https://wellecks.wordpress.com/2015/02/21/peering-into-the-black-box-visualizing-lambdamart/}
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#'
#' bst <- xgboost(data = agaricus.train$data, label = agaricus.train$label, max_depth = 15,
#'                  eta = 1, nthread = 2, nrounds = 30, objective = "binary:logistic",
#'                  min_child_weight = 50)
#'
#' p <- xgb.plot.multi.trees(model = bst, feature_names = colnames(agaricus.train$data),
#'                           features_keep = 3)
#' print(p)
#'
#' @export
xgb.plot.multi.trees <- function(model, feature_names = NULL, features_keep = 5, plot_width = NULL, plot_height = NULL, ...){
  check.deprecation(...)
  tree.matrix <- xgb.model.dt.tree(feature_names = feature_names, model = model)
  
  # first number of the path represents the tree, then the following numbers are related to the path to follow
  # root init
  root.nodes <- tree.matrix[stri_detect_regex(ID, "\\d+-0"), ID]
  tree.matrix[ID %in% root.nodes, abs.node.position:=root.nodes]
  
  precedent.nodes <- root.nodes
  
  while(tree.matrix[,sum(is.na(abs.node.position))] > 0) {
    yes.row.nodes <- tree.matrix[abs.node.position %in% precedent.nodes & !is.na(Yes)]
    no.row.nodes <- tree.matrix[abs.node.position %in% precedent.nodes & !is.na(No)]
    yes.nodes.abs.pos <- yes.row.nodes[, abs.node.position] %>% paste0("_0")
    no.nodes.abs.pos <- no.row.nodes[, abs.node.position] %>% paste0("_1")
    
    tree.matrix[ID %in% yes.row.nodes[, Yes], abs.node.position := yes.nodes.abs.pos]
    tree.matrix[ID %in% no.row.nodes[, No], abs.node.position := no.nodes.abs.pos]
    precedent.nodes <- c(yes.nodes.abs.pos, no.nodes.abs.pos)
  }
  
  tree.matrix[!is.na(Yes),Yes:= paste0(abs.node.position, "_0")]
  tree.matrix[!is.na(No),No:= paste0(abs.node.position, "_1")]
  
  
  remove.tree <- . %>% stri_replace_first_regex(pattern = "^\\d+-", replacement = "")
  
  tree.matrix[,`:=`(abs.node.position = remove.tree(abs.node.position),
                    Yes = remove.tree(Yes),
                    No = remove.tree(No))]
  
  nodes.dt <- tree.matrix[
        , .(Quality = sum(Quality))
        , by = .(abs.node.position, Feature)
      ][, .(Text = paste0(Feature[1:min(length(Feature), features_keep)],
                          " (",
                          format(Quality[1:min(length(Quality), features_keep)], digits=5),
                          ")") %>%
                   paste0(collapse = "\n"))
        , by = abs.node.position]
  
  edges.dt <- tree.matrix[Feature != "Leaf", .(abs.node.position, Yes)] %>%
    list(tree.matrix[Feature != "Leaf",.(abs.node.position, No)]) %>%
    rbindlist() %>%
    setnames(c("From", "To")) %>%
    .[, .N, .(From, To)] %>%
    .[, N:=NULL]
  
  nodes <- DiagrammeR::create_node_df(
    n = nrow(nodes.dt),
    label = nodes.dt[,Text]
  )
  
  edges <- DiagrammeR::create_edge_df(
    from = match(edges.dt[,From], nodes.dt[,abs.node.position]),
    to = match(edges.dt[,To], nodes.dt[,abs.node.position]),
    rel = "leading_to")
  
  graph <- DiagrammeR::create_graph(
      nodes_df = nodes,
      edges_df = edges,
      attr_theme = NULL
      ) %>%
    DiagrammeR::add_global_graph_attrs(
      attr_type = "graph",
      attr  = c("layout", "rankdir"),
      value = c("dot", "LR")
      ) %>%
    DiagrammeR::add_global_graph_attrs(
      attr_type = "node",
      attr  = c("color", "fillcolor", "style", "shape", "fontname"),
      value = c("DimGray", "beige", "filled", "rectangle", "Helvetica")
      ) %>%
    DiagrammeR::add_global_graph_attrs(
      attr_type = "edge",
      attr  = c("color", "arrowsize", "arrowhead", "fontname"),
      value = c("DimGray", "1.5", "vee", "Helvetica"))
  
  DiagrammeR::render_graph(graph, width = plot_width, height = plot_height)  
}

globalVariables(c(".N", "N", "From", "To", "Text", "Feature", "no.nodes.abs.pos",
                  "ID", "Yes", "No", "Tree", "yes.nodes.abs.pos", "abs.node.position"))
