#' Plot boosted trees
#'
#' Read a tree model text dump and plot the model.
#' 
#' @param feature_names Character vector used to overwrite the feature names
#'        of the model. The default (`NULL`) uses the original feature names.
#' @param model Object of class `xgb.Booster`.
#' @param trees An integer vector of tree indices that should be used. 
#'        The default (`NULL`) uses all trees.
#'        Useful, e.g., in multiclass classification to get only
#'        the trees of one class. *Important*: the tree index in XGBoost models
#'        is zero-based (e.g., use `trees = 0:2` for the first three trees).
#' @param plot_width,plot_height Width and height of the graph in pixels.
#'        The values are passed to [DiagrammeR::render_graph()].
#' @param render Should the graph be rendered or not? The default is `TRUE`.
#' @param show_node_id a logical flag for whether to show node id's in the graph.
#' @param ... currently not used.
#'
#' @details
#'
#' The content of each node is visualized like this:
#' - *Feature name*.
#' - *Cover:* The sum of second order gradients of training data.
#'   For the squared loss, this simply corresponds to the number of instances in the node.
#'   The deeper in the tree, the lower the value.
#' - *Gain* (for split nodes): Information gain metric of a split
#'        (corresponds to the importance of the node in the model).
#' - *Value* (for leaves): Margin value that the leaf may contribute to the prediction.
#' 
#' The tree root nodes also indicate the tree index (0-based).
#'
#' The "Yes" branches are marked by the "< split_value" label.
#' The branches also used for missing values are marked as bold
#' (as in "carrying extra capacity").
#'
#' This function uses [GraphViz](https://www.graphviz.org/) as DiagrammeR backend.
#' 
#' @return
#' The value depends on the `render` parameter:
#' - If `render = TRUE` (default): Rendered graph object which is an htmlwidget of 
#'   class `grViz`. Similar to "ggplot" objects, it needs to be printed when not 
#'   running from the command line.
#' - If `render = FALSE`: Graph object which is of DiagrammeR's class `dgr_graph`.
#'   This could be useful if one wants to modify some of the graph attributes
#'   before rendering the graph with [DiagrammeR::render_graph()].
#'
#' @examples
#' data(agaricus.train, package = "xgboost")
#'
#' bst <- xgboost(
#'   data = agaricus.train$data,
#'   label = agaricus.train$label,
#'   max_depth = 3,
#'   eta = 1,
#'   nthread = 2,
#'   nrounds = 2,
#'   objective = "binary:logistic"
#' )
#' 
#' # plot all the trees
#' xgb.plot.tree(model = bst)
#' 
#' # plot only the first tree and display the node ID:
#' xgb.plot.tree(model = bst, trees = 0, show_node_id = TRUE)
#'
#' \dontrun{
#' # Below is an example of how to save this plot to a file.
#' # Note that for export_graph() to work, the {DiagrammeRsvg}
#' # and {rsvg} packages must also be installed.
#' 
#' library(DiagrammeR)
#' 
#' gr <- xgb.plot.tree(model = bst, trees = 0:1, render = FALSE)
#' export_graph(gr, "tree.pdf", width = 1500, height = 1900)
#' export_graph(gr, "tree.png", width = 1500, height = 1900)
#' }
#'
#' @export
xgb.plot.tree <- function(feature_names = NULL, model = NULL, trees = NULL, plot_width = NULL, plot_height = NULL,
                          render = TRUE, show_node_id = FALSE, ...) {
  check.deprecation(...)
  if (!inherits(model, "xgb.Booster")) {
    stop("model: Has to be an object of class xgb.Booster")
  }

  if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
    stop("DiagrammeR package is required for xgb.plot.tree", call. = FALSE)
  }

  dt <- xgb.model.dt.tree(feature_names = feature_names, model = model, trees = trees)

  dt[, label := paste0(Feature, "\nCover: ", Cover, ifelse(Feature == "Leaf", "\nValue: ", "\nGain: "), Quality)]
  if (show_node_id)
    dt[, label := paste0(ID, ": ", label)]
  dt[Node == 0, label := paste0("Tree ", Tree, "\n", label)]
  dt[, shape := "rectangle"][Feature == "Leaf", shape := "oval"]
  dt[, filledcolor := "Beige"][Feature == "Leaf", filledcolor := "Khaki"]
  # in order to draw the first tree on top:
  dt <- dt[order(-Tree)]

  nodes <- DiagrammeR::create_node_df(
    n         = nrow(dt),
    ID        = dt$ID,
    label     = dt$label,
    fillcolor = dt$filledcolor,
    shape     = dt$shape,
    data      = dt$Feature,
    fontcolor = "black")

  if (nrow(dt[Feature != "Leaf"]) != 0) {
    edges <- DiagrammeR::create_edge_df(
      from  = match(rep(dt[Feature != "Leaf", c(ID)], 2), dt$ID),
      to    = match(dt[Feature != "Leaf", c(Yes, No)], dt$ID),
      label = c(
        dt[Feature != "Leaf", paste("<", Split)],
        rep("", nrow(dt[Feature != "Leaf"]))
      ),
      style = c(
        dt[Feature != "Leaf", ifelse(Missing == Yes, "bold", "solid")],
        dt[Feature != "Leaf", ifelse(Missing == No, "bold", "solid")]
      ),
      rel   = "leading_to")
  } else {
    edges <- NULL
  }

  graph <- DiagrammeR::create_graph(
      nodes_df = nodes,
      edges_df = edges,
      attr_theme = NULL
  )
  graph <- DiagrammeR::add_global_graph_attrs(
      graph = graph,
      attr_type = "graph",
      attr  = c("layout", "rankdir"),
      value = c("dot", "LR")
  )
  graph <- DiagrammeR::add_global_graph_attrs(
      graph = graph,
      attr_type = "node",
      attr  = c("color", "style", "fontname"),
      value = c("DimGray", "filled", "Helvetica")
  )
  graph <- DiagrammeR::add_global_graph_attrs(
      graph = graph,
      attr_type = "edge",
      attr  = c("color", "arrowsize", "arrowhead", "fontname"),
      value = c("DimGray", "1.5", "vee", "Helvetica")
  )

  if (!render) return(invisible(graph))

  DiagrammeR::render_graph(graph, width = plot_width, height = plot_height)
}

# Avoid error messages during CRAN check.
# The reason is that these variables are never declared
# They are mainly column names inferred by Data.table...
globalVariables(c("Feature", "ID", "Cover", "Quality", "Split", "Yes", "No", "Missing", ".", "shape", "filledcolor", "label"))
