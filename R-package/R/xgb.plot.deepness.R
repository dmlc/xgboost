#' Plot model tree depth
#'
#' Visualizes distributions related to the depth of tree leaves.
#' - `xgb.plot.deepness()` uses base R graphics, while
#' - `xgb.ggplot.deepness()` uses "ggplot2".
#'
#' @param model Either an `xgb.Booster` model, or the "data.table" returned
#'   by [xgb.model.dt.tree()].
#' @param which Which distribution to plot (see details).
#' @param plot Should the plot be shown? Default is `TRUE`.
#' @param ... Other parameters passed to [graphics::barplot()] or [graphics::plot()].
#'
#' @details
#'
#' When `which = "2x1"`, two distributions with respect to the leaf depth
#' are plotted on top of each other:
#' 1. The distribution of the number of leaves in a tree model at a certain depth.
#' 2. The distribution of the average weighted number of observations ("cover")
#'   ending up in leaves at a certain depth.
#'
#' Those could be helpful in determining sensible ranges of the `max_depth`
#' and `min_child_weight` parameters.
#'
#' When `which = "max.depth"` or `which = "med.depth"`, plots of either maximum or
#' median depth per tree with respect to the tree number are created.
#'
#' Finally, `which = "med.weight"` allows to see how
#' a tree's median absolute leaf weight changes through the iterations.
#'
#' These functions have been inspired by the blog post
#' <https://github.com/aysent/random-forest-leaf-visualization>.
#'
#' @return
#' The return value of the two functions is as follows:
#' - `xgb.plot.deepness()`: A "data.table" (invisibly).
#'   Each row corresponds to a terminal leaf in the model. It contains its information
#'   about depth, cover, and weight (used in calculating predictions).
#'   If `plot = TRUE`, also a plot is shown.
#' - `xgb.ggplot.deepness()`: When `which = "2x1"`, a list of two "ggplot" objects,
#'   and a single "ggplot" object otherwise.
#'
#' @seealso [xgb.train()] and [xgb.model.dt.tree()].
#'
#' @examples
#'
#' data(agaricus.train, package = "xgboost")
#' ## Keep the number of threads to 2 for examples
#' nthread <- 2
#' data.table::setDTthreads(nthread)
#'
#' ## Change max_depth to a higher number to get a more significant result
#' model <- xgboost(
#'   agaricus.train$data, factor(agaricus.train$label),
#'   nrounds = 50,
#'   max_depth = 6,
#'   nthreads = nthread,
#'   subsample = 0.5,
#'   min_child_weight = 2
#' )
#'
#' xgb.plot.deepness(model)
#' xgb.ggplot.deepness(model)
#'
#' xgb.plot.deepness(
#'   model, which = "max.depth", pch = 16, col = rgb(0, 0, 1, 0.3), cex = 2
#' )
#'
#' xgb.plot.deepness(
#'   model, which = "med.weight", pch = 16, col = rgb(0, 0, 1, 0.3), cex = 2
#' )
#'
#' @rdname xgb.plot.deepness
#' @export
xgb.plot.deepness <- function(model = NULL, which = c("2x1", "max.depth", "med.depth", "med.weight"),
                              plot = TRUE, ...) {

  if (!(inherits(model, "xgb.Booster") || is.data.table(model)))
    stop("model: Has to be either an xgb.Booster model generaged by the xgb.train function\n",
         "or a data.table result of the xgb.importance function")

  if (!requireNamespace("igraph", quietly = TRUE))
    stop("igraph package is required for plotting the graph deepness.", call. = FALSE)

  which <- match.arg(which)

  dt_tree <- model
  if (inherits(model, "xgb.Booster"))
    dt_tree <- xgb.model.dt.tree(model = model)

  if (!all(c("Feature", "Tree", "ID", "Yes", "No", "Cover") %in% colnames(dt_tree)))
    stop("Model tree columns are not as expected!\n",
         "  Note that this function works only for tree models.")

  dt_depths <- merge(get.leaf.depth(dt_tree), dt_tree[, .(ID, Cover, Weight = Gain)], by = "ID")
  setkeyv(dt_depths, c("Tree", "ID"))
  # count by depth levels, and also calculate average cover at a depth
  dt_summaries <- dt_depths[, .(.N, Cover = mean(Cover)), Depth]
  setkey(dt_summaries, "Depth")

  if (plot) {
    if (which == "2x1") {
      op <- par(no.readonly = TRUE)
      par(mfrow = c(2, 1),
          oma = c(3, 1, 3, 1) + 0.1,
          mar = c(1, 4, 1, 0) + 0.1)

      dt_summaries[, barplot(N, border = NA, ylab = 'Number of leafs', ...)]

      dt_summaries[, barplot(Cover, border = NA, ylab = "Weighted cover", names.arg = Depth, ...)]

      title("Model complexity", xlab = "Leaf depth", outer = TRUE, line = 1)
      par(op)
    } else if (which == "max.depth") {
      dt_depths[, max(Depth), Tree][
                , plot(jitter(V1, amount = 0.1) ~ Tree, ylab = 'Max tree leaf depth', xlab = "tree #", ...)]
    } else if (which == "med.depth") {
      dt_depths[, median(as.numeric(Depth)), Tree][
                , plot(jitter(V1, amount = 0.1) ~ Tree, ylab = 'Median tree leaf depth', xlab = "tree #", ...)]
    } else if (which == "med.weight") {
      dt_depths[, median(abs(Weight)), Tree][
                , plot(V1 ~ Tree, ylab = 'Median absolute leaf weight', xlab = "tree #", ...)]
    }
  }
  invisible(dt_depths)
}

# Extract path depths from root to leaf
# from data.table containing the nodes and edges of the trees.
# internal utility function
get.leaf.depth <- function(dt_tree) {
  # extract tree graph's edges
  dt_edges <- rbindlist(list(
      dt_tree[Feature != "Leaf", .(ID, To = Yes, Tree)],
      dt_tree[Feature != "Leaf", .(ID, To = No, Tree)]
    ))
  # whether "To" is a leaf:
  dt_edges <-
    merge(dt_edges,
          dt_tree[Feature == "Leaf", .(ID, Leaf = TRUE)],
          all.x = TRUE, by.x = "To", by.y = "ID")
  dt_edges[is.na(Leaf), Leaf := FALSE]

  dt_edges[, {
    graph <- igraph::graph_from_data_frame(.SD[, .(ID, To)])
    # min(ID) in a tree is a root node
    paths_tmp <- igraph::shortest_paths(graph, from = min(ID), to = To[Leaf == TRUE])
    # list of paths to each leaf in a tree
    paths <- lapply(paths_tmp$vpath, names)
    # combine into a resulting path lengths table for a tree
    data.table(Depth = lengths(paths), ID = To[Leaf == TRUE])
  }, by = Tree]
}

# Avoid error messages during CRAN check.
# The reason is that these variables are never declared
# They are mainly column names inferred by Data.table...
globalVariables(
  c(
    ".N", "N", "Depth", "Gain", "Cover", "Tree", "ID", "Yes", "No", "Feature", "Leaf", "Weight"
  )
)
