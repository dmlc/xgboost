# ggplot backend for the xgboost plotting facilities


#' @rdname xgb.plot.importance
#' @export
xgb.ggplot.importance <- function(importance_matrix = NULL, top_n = NULL, measure = NULL, 
                                  rel_to_first = FALSE, n_clusters = c(1:10), ...) {
  
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
                           height = 0.15, alpha=0.4, size=3, stroke=0) + 
      ggplot2::xlab("tree #") +
      ggplot2::ylab("Max tree leaf depth")
    return(p)
    
  } else if (which == "med.depth") {
    p <-
      ggplot2::ggplot(dt_depths[, median(as.numeric(Depth)), Tree]) +
      ggplot2::geom_jitter(ggplot2::aes(x = Tree, y = V1),
                           height = 0.15, alpha=0.4, size=3, stroke=0) + 
      ggplot2::xlab("tree #") +
      ggplot2::ylab("Median tree leaf depth")
    return(p)

  } else if (which == "med.weight") {
    p <-
      ggplot2::ggplot(dt_depths[, median(abs(Weight)), Tree]) +
      ggplot2::geom_point(ggplot2::aes(x = Tree, y = V1),
                          alpha=0.4, size=3, stroke=0) + 
      ggplot2::xlab("tree #") +
      ggplot2::ylab("Median absolute leaf weight")
    return(p)
  }
}

# Plot multiple ggplot graph aligned by rows and columns.
# ... the plots
# cols number of columns
# internal utility function
multiplot <- function(..., cols = 1) {
  plots <- list(...)
  num_plots = length(plots)
  
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
  "element_blank", "element_text", "V1", "Weight"
))
