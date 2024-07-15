#' Plot boosted trees
#'
#' Read a tree model text dump and plot the model.
#'
#' @param model Object of class `xgb.Booster`. If it contains feature names (they can be set through
#'        \link{setinfo}), they will be used in the output from this function.
#' @param trees An integer vector of tree indices that should be used.
#'        The default (`NULL`) uses all trees.
#'        Useful, e.g., in multiclass classification to get only
#'        the trees of one class. *Important*: the tree index in XGBoost models
#'        is zero-based (e.g., use `trees = 0:2` for the first three trees).
#' @param plot_width,plot_height Width and height of the graph in pixels.
#'        The values are passed to [DiagrammeR::render_graph()].
#' @param render Should the graph be rendered or not? The default is `TRUE`.
#' @param show_node_id a logical flag for whether to show node id's in the graph.
#' @param style Style to use for the plot. Options are:\itemize{
#' \item `"xgboost"`: will use the plot style defined in the core XGBoost library,
#' which is shared between different interfaces through the 'dot' format. This
#' style was not available before version 2.1.0 in R. It always plots the trees
#' vertically (from top to bottom).
#' \item `"R"`: will use the style defined from XGBoost's R interface, which predates
#' the introducition of the standardized style from the core library. It might plot
#' the trees horizontally (from left to right).
#' }
#'
#' Note that `style="xgboost"` is only supported when all of the following conditions are met:\itemize{
#' \item Only a single tree is being plotted.
#' \item Node IDs are not added to the graph.
#' \item The graph is being returned as `htmlwidget` (`render=TRUE`).
#' }
#' @param ... currently not used.
#'
#' @details
#'
#' When using `style="xgboost"`, the content of each node is visualized as follows:
#' - For non-terminal nodes, it will display the split condition (number or name if
#'   available, and the condition that would decide to which node to go next).
#' - Those nodes will be connected to their children by arrows that indicate whether the
#'   branch corresponds to the condition being met or not being met.
#' - Terminal (leaf) nodes contain the margin to add when ending there.
#'
#' When using `style="R"`, the content of each node is visualized like this:
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
#' bst <- xgb.train(
#'   data = xgb.DMatrix(agaricus.train$data, agaricus.train$label),
#'   max_depth = 3,
#'   eta = 1,
#'   nthread = 2,
#'   nrounds = 2,
#'   objective = "binary:logistic"
#' )
#'
#' # plot the first tree, using the style from xgboost's core library
#' # (this plot should look identical to the ones generated from other
#' # interfaces like the python package for xgboost)
#' xgb.plot.tree(model = bst, trees = 1, style = "xgboost")
#'
#' # plot all the trees
#' xgb.plot.tree(model = bst, trees = NULL)
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
xgb.plot.tree <- function(model = NULL, trees = NULL, plot_width = NULL, plot_height = NULL,
                          render = TRUE, show_node_id = FALSE, style = c("R", "xgboost"), ...) {
  check.deprecation(...)
  if (!inherits(model, "xgb.Booster")) {
    stop("model: Has to be an object of class xgb.Booster")
  }

  if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
    stop("DiagrammeR package is required for xgb.plot.tree", call. = FALSE)
  }

  style <- as.character(head(style, 1L))
  stopifnot(style %in% c("R", "xgboost"))
  if (style == "xgboost") {
    if (NROW(trees) != 1L || !render || show_node_id) {
      stop("style='xgboost' is only supported for single, rendered tree, without node IDs.")
    }

    txt <- xgb.dump(model, dump_format = "dot")
    return(DiagrammeR::grViz(txt[[trees + 1]], width = plot_width, height = plot_height))
  }

  dt <- xgb.model.dt.tree(model = model, trees = trees)

  dt[, label := paste0(Feature, "\nCover: ", Cover, ifelse(Feature == "Leaf", "\nValue: ", "\nGain: "), Gain)]
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
globalVariables(c("Feature", "ID", "Cover", "Gain", "Split", "Yes", "No", "Missing", ".", "shape", "filledcolor", "label"))
