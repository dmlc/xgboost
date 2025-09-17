#' Plot boosted trees
#'
#' Read a tree model text dump and plot the model.
#'
#' @details
#' The content of each node is visualized as follows:
#' - For non-terminal nodes, it will display the split condition (number or name
#'   if available, and the condition that would decide to which node to go
#'   next).
#' - Those nodes will be connected to their children by arrows that indicate
#'   whether the branch corresponds to the condition being met or not being met.
#' - Terminal (leaf) nodes contain the margin to add when ending there.
#'
#' The "Yes" branches are marked by the "< split_value" label.
#' The branches also used for missing values are marked as bold
#' (as in "carrying extra capacity").
#'
#' This function uses [GraphViz](https://www.graphviz.org/) as DiagrammeR
#' backend.
#'
#' @param model Object of class `xgb.Booster`. If it contains feature names
#'   (they can be set through [setinfo()], they will be used in the
#'   output from this function.
#' @param tree_idx An integer of the tree index that should be used. This
#'   is an 1-based index.
#' @param plot_width,plot_height Width and height of the graph in pixels.
#'   The values are passed to `DiagrammeR::render_graph()`.
#' @param with_stats Whether to dump some additional statistics about the
#'   splits.  When this option is on, the model dump contains two additional
#'   values: gain is the approximate loss function gain we get in each split;
#'   cover is the sum of second order gradient in each node.
#' @inheritParams xgb.train
#' @return
#'
#' Rendered graph object which is an htmlwidget of ' class `grViz`. Similar to
#' "ggplot" objects, it needs to be printed when not running from the command
#' line.
#'
#' @examples
#' data("ToothGrowth")
#' x <- ToothGrowth[, c("len", "dose")]
#' y <- ToothGrowth$supp
#' model <- xgboost(
#'   x, y,
#'   nthreads = 1L,
#'   nrounds = 3L,
#'   max_depth = 3L
#' )
#'
#' # plot the first tree
#' xgb.plot.tree(model, tree_idx = 1)
#'
#' # Below is an example of how to save this plot to a file.
#' if (require("DiagrammeR") && require("htmlwidgets")) {
#'   fname <- file.path(tempdir(), "plot.html'")
#'   gr <- xgb.plot.tree(model, tree_idx = 1)
#'   htmlwidgets::saveWidget(gr, fname)
#' }
#' @export
xgb.plot.tree <- function(model,
                          tree_idx = 1,
                          plot_width = NULL,
                          plot_height = NULL,
                          with_stats = FALSE, ...) {
  check.deprecation(deprecated_plottree_params, match.call(), ...)
  if (!inherits(model, "xgb.Booster")) {
    stop("model has to be an object of the class xgb.Booster")
  }
  if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
    stop("The DiagrammeR package is required for xgb.plot.tree", call. = FALSE)
  }

  txt <- xgb.dump(model, dump_format = "dot", with_stats = with_stats)
  DiagrammeR::grViz(
    txt[[tree_idx]], width = plot_width, height = plot_height
  )
}
