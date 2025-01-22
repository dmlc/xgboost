#' Project all trees on one tree
#'
#' Visualization of the ensemble of trees as a single collective unit.
#'
#' Note that this function does not work with models that were fitted to
#' categorical data.
#' @details
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
#' `features_keep` first features (based on the Gain per feature measure).
#'
#' This function is inspired by this blog post:
#' <https://wellecks.wordpress.com/2015/02/21/peering-into-the-black-box-visualizing-lambdamart/>
#'
#' @inheritParams xgb.plot.tree
#' @param features_keep Number of features to keep in each position of the multi trees,
#'   by default 5.
#' @param render Should the graph be rendered or not? The default is `TRUE`.
#' @inherit xgb.plot.tree return
#'
#' @examples
#'
#' data(agaricus.train, package = "xgboost")
#'
#' ## Keep the number of threads to 2 for examples
#' nthread <- 2
#' data.table::setDTthreads(nthread)
#'
#' model <- xgboost(
#'   agaricus.train$data, factor(agaricus.train$label),
#'   nrounds = 30,
#'   verbosity = 0L,
#'   nthreads = nthread,
#'   max_depth = 15,
#'   learning_rate = 1,
#'   min_child_weight = 50
#' )
#'
#' p <- xgb.plot.multi.trees(model, features_keep = 3)
#' print(p)
#'
#' # Below is an example of how to save this plot to a file.
#' if (require("DiagrammeR") && require("DiagrammeRsvg") && require("rsvg")) {
#'   fname <- file.path(tempdir(), "tree.pdf")
#'   gr <- xgb.plot.multi.trees(model, features_keep = 3, render = FALSE)
#'   export_graph(gr, fname, width = 1500, height = 600)
#' }
#' @export
xgb.plot.multi.trees <- function(model, features_keep = 5, plot_width = NULL, plot_height = NULL,
                                 render = TRUE, ...) {
  check.deprecation(deprecated_multitrees_params, match.call(), ...)
  if (!requireNamespace("DiagrammeR", quietly = TRUE)) {
    stop("DiagrammeR is required for xgb.plot.multi.trees")
  }
  if (xgb.has_categ_features(model)) {
    stop(
      "Cannot use 'xgb.plot.multi.trees' for models with categorical features.",
      " Try 'xgb.plot.tree' instead."
    )
  }
  tree.matrix <- xgb.model.dt.tree(model = model)

  # first number of the path represents the tree, then the following numbers are related to the path to follow
  # root init
  root.nodes <- tree.matrix[Node == 0, ID]
  tree.matrix[ID %in% root.nodes, abs.node.position := root.nodes]

  precedent.nodes <- root.nodes

  while (tree.matrix[, sum(is.na(abs.node.position))] > 0) {
    yes.row.nodes <- tree.matrix[abs.node.position %in% precedent.nodes & !is.na(Yes)]
    no.row.nodes <- tree.matrix[abs.node.position %in% precedent.nodes & !is.na(No)]
    yes.nodes.abs.pos <- paste0(yes.row.nodes[, abs.node.position], "_0")
    no.nodes.abs.pos <- paste0(no.row.nodes[, abs.node.position], "_1")

    tree.matrix[ID %in% yes.row.nodes[, Yes], abs.node.position := yes.nodes.abs.pos]
    tree.matrix[ID %in% no.row.nodes[, No], abs.node.position := no.nodes.abs.pos]
    precedent.nodes <- c(yes.nodes.abs.pos, no.nodes.abs.pos)
  }

  tree.matrix[!is.na(Yes), Yes := paste0(abs.node.position, "_0")]
  tree.matrix[!is.na(No), No := paste0(abs.node.position, "_1")]

  for (nm in c("abs.node.position", "Yes", "No"))
    data.table::set(tree.matrix, j = nm, value = sub("^\\d+-", "", tree.matrix[[nm]]))

  nodes.dt <- tree.matrix[
        , .(Gain = sum(Gain))
        , by = .(abs.node.position, Feature)
      ][, .(Text = paste0(
              paste0(
                Feature[seq_len(min(length(Feature), features_keep))],
                " (",
                format(Gain[seq_len(min(length(Gain), features_keep))], digits = 5),
                ")"
              ),
              collapse = "\n"
            )
          )
        , by = abs.node.position
      ]

  edges.dt <- data.table::rbindlist(
    l = list(
      tree.matrix[Feature != "Leaf", .(From = abs.node.position, To = Yes)],
      tree.matrix[Feature != "Leaf", .(From = abs.node.position, To = No)]
    )
  )
  edges.dt <- edges.dt[, .N, .(From, To)]
  edges.dt[, N := NULL]

  nodes <- DiagrammeR::create_node_df(
    n = nrow(nodes.dt),
    label = nodes.dt[, Text]
  )

  edges <- DiagrammeR::create_edge_df(
    from = match(edges.dt[, From], nodes.dt[, abs.node.position]),
    to = match(edges.dt[, To], nodes.dt[, abs.node.position]),
    rel = "leading_to")

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
      attr  = c("color", "fillcolor", "style", "shape", "fontname"),
      value = c("DimGray", "beige", "filled", "rectangle", "Helvetica")
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

globalVariables(c(".N", "N", "From", "To", "Text", "Feature", "no.nodes.abs.pos",
                  "ID", "Yes", "No", "Tree", "yes.nodes.abs.pos", "abs.node.position"))
