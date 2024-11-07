#' Plot boosted trees
#'
#' Read a tree model text dump and plot the model.
#'
#' @details
#' When using `style="xgboost"`, the content of each node is visualized as follows:
#' - For non-terminal nodes, it will display the split condition (number or name if
#'   available, and the condition that would decide to which node to go next).
#' - Those nodes will be connected to their children by arrows that indicate whether the
#'   branch corresponds to the condition being met or not being met.
#' - Terminal (leaf) nodes contain the margin to add when ending there.
#'
#' The tree root nodes also indicate the tree index (0-based).
#'
#' The "Yes" branches are marked by the "< split_value" label.
#' The branches also used for missing values are marked as bold
#' (as in "carrying extra capacity").
#'
#' This function uses [GraphViz](https://www.graphviz.org/) as DiagrammeR backend.
#'
#' @param model Object of class `xgb.Booster`. If it contains feature names (they can be set through
#'   [setinfo()], they will be used in the output from this function.
#' @param trees An integer vector of tree indices that should be used.
#'   The default (`NULL`) uses all trees.
#'   Useful, e.g., in multiclass classification to get only
#'   the trees of one class. *Important*: the tree index in XGBoost models
#'   is zero-based (e.g., use `trees = 0:2` for the first three trees).
#' @param plot_width,plot_height Width and height of the graph in pixels.
#'   The values are passed to `DiagrammeR::render_graph()`.
#' @param ... Currently not used.
#' @return
#' Rendered graph object which is an htmlwidget of ' class `grViz`. Similar to "ggplot" objects,
#' it needs to be printed when not running from the command line.
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
#' # plot the first tree
#' xgb.plot.tree(model = bst, trees = 1)
#'
#'
#' \dontrun{
#' # Below is an example of how to save this plot to a file.
#' # Note that for export_graph() to work, the {DiagrammeRsvg}
#' # and {rsvg} packages must also be installed.
#'
#' library(DiagrammeR)
#'
#' gr <- xgb.plot.tree(model = bst, trees = 0:1)
#' export_graph(gr, "tree.pdf", width = 1500, height = 1900)
#' export_graph(gr, "tree.png", width = 1500, height = 1900)
#' }
#'
#' @export
xgb.plot.tree <- function(model = NULL, trees = NULL, plot_width = NULL, plot_height = NULL, with_stats = FALSE, ...) {
  check.deprecation(...)
  if (!inherits(model, "xgb.Booster")) {
    stop("model: Has to be an object of class xgb.Booster")
  }
  if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
    stop("DiagrammeR package is required for xgb.plot.tree", call. = FALSE)
  }

  if (NROW(trees) != 1L) {
    stop("Plot a single tree at a time.")
  }

  txt <- xgb.dump(model, dump_format = "dot", with_stats = with_stats)

  DiagrammeR::grViz(txt[[trees + 1]], width = plot_width, height = plot_height)
}
