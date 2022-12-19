# ggplot backend for the xgboost plotting facilities


#' @rdname xgb.plot.importance
#' @export
xgb.ggplot.importance <- function(importance_matrix = NULL, top_n = NULL, measure = NULL,
                                  rel_to_first = FALSE, n_clusters = seq_len(10), ...) {

  importance_matrix <- xgb.plot.importance(importance_matrix, top_n = top_n, measure = measure,
                                           rel_to_first = rel_to_first, plot = FALSE, ...)
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required", call. = FALSE)
  }
  if (!requireNamespace("Ckmeans.1d.dp", quietly = TRUE)) {
    stop("Ckmeans.1d.dp package is required", call. = FALSE)
  }

  clusters <- suppressWarnings(
    Ckmeans.1d.dp::Ckmeans.1d.dp(importance_matrix$Importance, n_clusters)
  )
  importance_matrix[, Cluster := as.character(clusters$cluster)]

  plot <-
    ggplot2::ggplot(importance_matrix,
                    ggplot2::aes(x = factor(Feature, levels = rev(Feature)), y = Importance, width = 0.5),
                    environment = environment()) +
    ggplot2::geom_bar(ggplot2::aes(fill = Cluster), stat = "identity", position = "identity") +
    ggplot2::coord_flip() +
    ggplot2::xlab("Features") +
    ggplot2::ggtitle("Feature importance") +
    ggplot2::theme(plot.title = ggplot2::element_text(lineheight = .9, face = "bold"),
                   panel.grid.major.y = ggplot2::element_blank())
  return(plot)
}


#' @rdname xgb.plot.deepness
#' @export
xgb.ggplot.deepness <- function(model = NULL, which = c("2x1", "max.depth", "med.depth", "med.weight")) {

  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("ggplot2 package is required for plotting the graph deepness.", call. = FALSE)

  which <- match.arg(which)

  dt_depths <- xgb.plot.deepness(model = model, plot = FALSE)
  dt_summaries <- dt_depths[, .(.N, Cover = mean(Cover)), Depth]
  setkey(dt_summaries, 'Depth')

  if (which == "2x1") {
    p1 <-
      ggplot2::ggplot(dt_summaries) +
      ggplot2::geom_bar(ggplot2::aes(x = Depth, y = N), stat = "Identity") +
      ggplot2::xlab("") +
      ggplot2::ylab("Number of leafs") +
      ggplot2::ggtitle("Model complexity") +
      ggplot2::theme(
        plot.title = ggplot2::element_text(lineheight = 0.9, face = "bold"),
        panel.grid.major.y = ggplot2::element_blank(),
        axis.ticks = ggplot2::element_blank(),
        axis.text.x = ggplot2::element_blank()
      )

    p2 <-
      ggplot2::ggplot(dt_summaries) +
      ggplot2::geom_bar(ggplot2::aes(x = Depth, y = Cover), stat = "Identity") +
      ggplot2::xlab("Leaf depth") +
      ggplot2::ylab("Weighted cover")

    multiplot(p1, p2, cols = 1)
    return(invisible(list(p1, p2)))

  } else if (which == "max.depth") {
    p <-
      ggplot2::ggplot(dt_depths[, max(Depth), Tree]) +
      ggplot2::geom_jitter(ggplot2::aes(x = Tree, y = V1),
                           height = 0.15, alpha = 0.4, size = 3, stroke = 0) +
      ggplot2::xlab("tree #") +
      ggplot2::ylab("Max tree leaf depth")
    return(p)

  } else if (which == "med.depth") {
    p <-
      ggplot2::ggplot(dt_depths[, median(as.numeric(Depth)), Tree]) +
      ggplot2::geom_jitter(ggplot2::aes(x = Tree, y = V1),
                           height = 0.15, alpha = 0.4, size = 3, stroke = 0) +
      ggplot2::xlab("tree #") +
      ggplot2::ylab("Median tree leaf depth")
    return(p)

  } else if (which == "med.weight") {
    p <-
      ggplot2::ggplot(dt_depths[, median(abs(Weight)), Tree]) +
      ggplot2::geom_point(ggplot2::aes(x = Tree, y = V1),
                          alpha = 0.4, size = 3, stroke = 0) +
      ggplot2::xlab("tree #") +
      ggplot2::ylab("Median absolute leaf weight")
    return(p)
  }
}

#' @rdname xgb.plot.shap.summary
#' @export
xgb.ggplot.shap.summary <- function(data, shap_contrib = NULL, features = NULL, top_n = 10, model = NULL,
                                    trees = NULL, target_class = NULL, approxcontrib = FALSE, subsample = NULL) {
  data_list <- xgb.shap.data(
    data = data,
    shap_contrib = shap_contrib,
    features = features,
    top_n = top_n,
    model = model,
    trees = trees,
    target_class = target_class,
    approxcontrib = approxcontrib,
    subsample = subsample,
    max_observations = 10000  # 10,000 samples per feature.
  )
  p_data <- prepare.ggplot.shap.data(data_list, normalize = TRUE)
  # Reverse factor levels so that the first level is at the top of the plot
  p_data[, "feature" := factor(feature, rev(levels(feature)))]
  p <- ggplot2::ggplot(p_data, ggplot2::aes(x = feature, y = p_data$shap_value, colour = p_data$feature_value)) +
    ggplot2::geom_jitter(alpha = 0.5, width = 0.1) +
    ggplot2::scale_colour_viridis_c(limits = c(-3, 3), option = "plasma", direction = -1) +
    ggplot2::geom_abline(slope = 0, intercept = 0, colour = "darkgrey") +
    ggplot2::coord_flip()

  p
}

#' Combine and melt feature values and SHAP contributions for sample
#' observations.
#'
#' Conforms to data format required for ggplot functions.
#'
#' Internal utility function.
#'
#' @param data_list List containing 'data' and 'shap_contrib' returned by
#'   \code{xgb.shap.data()}.
#' @param normalize Whether to standardize feature values to have mean 0 and
#'   standard deviation 1 (useful for comparing multiple features on the same
#'   plot). Default \code{FALSE}.
#'
#' @return A data.table containing the observation ID, the feature name, the
#'   feature value (normalized if specified), and the SHAP contribution value.
prepare.ggplot.shap.data <- function(data_list, normalize = FALSE) {
  data <- data_list[["data"]]
  shap_contrib <- data_list[["shap_contrib"]]

  data <- data.table::as.data.table(as.matrix(data))
  if (normalize) {
    data[, (names(data)) := lapply(.SD, normalize)]
  }
  data[, "id" := seq_len(nrow(data))]
  data_m <- data.table::melt.data.table(data, id.vars = "id", variable.name = "feature", value.name = "feature_value")

  shap_contrib <- data.table::as.data.table(as.matrix(shap_contrib))
  shap_contrib[, "id" := seq_len(nrow(shap_contrib))]
  shap_contrib_m <- data.table::melt.data.table(shap_contrib, id.vars = "id", variable.name = "feature", value.name = "shap_value")

  p_data <- data.table::merge.data.table(data_m, shap_contrib_m, by = c("id", "feature"))

  p_data
}

#' Scale feature value to have mean 0, standard deviation 1
#'
#' This is used to compare multiple features on the same plot.
#' Internal utility function
#'
#' @param x Numeric vector
#'
#' @return Numeric vector with mean 0 and sd 1.
normalize <- function(x) {
  loc <- mean(x, na.rm = TRUE)
  scale <- stats::sd(x, na.rm = TRUE)

  (x - loc) / scale
}

# Plot multiple ggplot graph aligned by rows and columns.
# ... the plots
# cols number of columns
# internal utility function
multiplot <- function(..., cols = 1) {
  plots <- list(...)
  num_plots <- length(plots)

  layout <- matrix(seq(1, cols * ceiling(num_plots / cols)),
                   ncol = cols, nrow = ceiling(num_plots / cols))

  if (num_plots == 1) {
    print(plots[[1]])
  } else {
    grid::grid.newpage()
    grid::pushViewport(grid::viewport(layout = grid::grid.layout(nrow(layout), ncol(layout))))
    for (i in 1:num_plots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.table(which(layout == i, arr.ind = TRUE))

      print(
        plots[[i]], vp = grid::viewport(
          layout.pos.row = matchidx$row,
          layout.pos.col = matchidx$col
        )
      )
    }
  }
}

globalVariables(c(
  "Cluster", "ggplot", "aes", "geom_bar", "coord_flip", "xlab", "ylab", "ggtitle", "theme",
  "element_blank", "element_text", "V1", "Weight", "feature"
))
