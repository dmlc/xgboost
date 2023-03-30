#' Plot a boosted tree model
#'
#' Read a tree model text dump and plot the model.
#'
#' @param feature_names names of each feature as a \code{character} vector.
#' @param model produced by the \code{xgb.train} function.
#' @param trees an integer vector of tree indices that should be visualized.
#'          If set to \code{NULL}, all trees of the model are included.
#'          IMPORTANT: the tree index in xgboost model is zero-based
#'          (e.g., use \code{trees = 0:2} for the first 3 trees in a model).
#' @param plot_width  the width of the diagram in pixels.
#' @param plot_height	the height of the diagram in pixels.
#' @param render a logical flag for whether the graph should be rendered (see Value).
#' @param show_node_id a logical flag for whether to show node id's in the graph.
#' @param ... currently not used.
#'
#' @details
#'
#' The content of each node is organised that way:
#'
#' \itemize{
#'  \item Feature name.
#'  \item \code{Cover}: The sum of second order gradient of training data classified to the leaf.
#'        If it is square loss, this simply corresponds to the number of instances seen by a split
#'        or collected by a leaf during training.
#'        The deeper in the tree a node is, the lower this metric will be.
#'  \item \code{Gain} (for split nodes): the information gain metric of a split
#'        (corresponds to the importance of the node in the model).
#'  \item \code{Value} (for leafs): the margin value that the leaf may contribute to prediction.
#' }
#' The tree root nodes also indicate the Tree index (0-based).
#'
#' The "Yes" branches are marked by the "< split_value" label.
#' The branches that also used for missing values are marked as bold
#' (as in "carrying extra capacity").
#'
#' This function uses \href{https://www.graphviz.org/}{GraphViz} as a backend of DiagrammeR.
#'
#' @return
#'
#' When \code{render = TRUE}:
#' returns a rendered graph object which is an \code{htmlwidget} of class \code{grViz}.
#' Similar to ggplot objects, it needs to be printed to see it when not running from command line.
#'
#' When \code{render = FALSE}:
#' silently returns a graph object which is of DiagrammeR's class \code{dgr_graph}.
#' This could be useful if one wants to modify some of the graph attributes
#' before rendering the graph with \code{\link[DiagrammeR]{render_graph}}.
#'
#' @examples
#' data(agaricus.train, package='xgboost')
#'
#' bst <- xgboost(data = agaricus.train$data, label = agaricus.train$label, max_depth = 3,
#'                eta = 1, nthread = 2, nrounds = 2,objective = "binary:logistic")
#' # plot all the trees
#' xgb.plot.tree(model = bst)
#' # plot only the first tree and display the node ID:
#' xgb.plot.tree(model = bst, trees = 0, show_node_id = TRUE)
#'
#' \dontrun{
#' # Below is an example of how to save this plot to a file.
#' # Note that for `export_graph` to work, the DiagrammeRsvg and rsvg packages must also be installed.
#' library(DiagrammeR)
#' gr <- xgb.plot.tree(model=bst, trees=0:1, render=FALSE)
#' export_graph(gr, 'tree.pdf', width=1500, height=1900)
#' export_graph(gr, 'tree.png', width=1500, height=1900)
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
