#' Plot feature importance as a bar graph
#'
#' Represents previously calculated feature importance as a bar graph.
#' \code{xgb.plot.importance} uses base R graphics, while \code{xgb.ggplot.importance} uses the ggplot backend.
#'
#' @param importance_matrix a \code{data.table} returned by \code{\link{xgb.importance}}.
#' @param top_n maximal number of top features to include into the plot.
#' @param measure the name of importance measure to plot.
#'        When \code{NULL}, 'Gain' would be used for trees and 'Weight' would be used for gblinear.
#' @param rel_to_first whether importance values should be represented as relative to the highest ranked feature.
#'        See Details.
#' @param left_margin (base R barplot) allows to adjust the left margin size to fit feature names.
#'        When it is NULL, the existing \code{par('mar')} is used.
#' @param cex (base R barplot) passed as \code{cex.names} parameter to \code{barplot}.
#' @param plot (base R barplot) whether a barplot should be produced.
#'        If FALSE, only a data.table is returned.
#' @param n_clusters (ggplot only) a \code{numeric} vector containing the min and the max range
#'        of the possible number of clusters of bars.
#' @param ... other parameters passed to \code{barplot} (except horiz, border, cex.names, names.arg, and las).
#'
#' @details
#' The graph represents each feature as a horizontal bar of length proportional to the importance of a feature.
#' Features are shown ranked in a decreasing importance order.
#' It works for importances from both \code{gblinear} and \code{gbtree} models.
#'
#' When \code{rel_to_first = FALSE}, the values would be plotted as they were in \code{importance_matrix}.
#' For gbtree model, that would mean being normalized to the total of 1
#' ("what is feature's importance contribution relative to the whole model?").
#' For linear models, \code{rel_to_first = FALSE} would show actual values of the coefficients.
#' Setting \code{rel_to_first = TRUE} allows to see the picture from the perspective of
#' "what is feature's importance contribution relative to the most important feature?"
#'
#' The ggplot-backend method also performs 1-D clustering of the importance values,
#' with bar colors corresponding to different clusters that have somewhat similar importance values.
#'
#' @return
#' The \code{xgb.plot.importance} function creates a \code{barplot} (when \code{plot=TRUE})
#' and silently returns a processed data.table with \code{n_top} features sorted by importance.
#'
#' The \code{xgb.ggplot.importance} function returns a ggplot graph which could be customized afterwards.
#' E.g., to change the title of the graph, add \code{+ ggtitle("A GRAPH NAME")} to the result.
#'
#' @seealso
#' \code{\link[graphics]{barplot}}.
#'
#' @examples
#' data(agaricus.train)
#'
#' bst <- xgboost(data = agaricus.train$data, label = agaricus.train$label, max_depth = 3,
#'                eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
#'
#' importance_matrix <- xgb.importance(colnames(agaricus.train$data), model = bst)
#'
#' xgb.plot.importance(importance_matrix, rel_to_first = TRUE, xlab = "Relative importance")
#'
#' (gg <- xgb.ggplot.importance(importance_matrix, measure = "Frequency", rel_to_first = TRUE))
#' gg + ggplot2::ylab("Frequency")
#'
#' @rdname xgb.plot.importance
#' @export
xgb.plot.importance <- function(importance_matrix = NULL, top_n = NULL, measure = NULL,
                                rel_to_first = FALSE, left_margin = 10, cex = NULL, plot = TRUE, ...) {
  check.deprecation(...)
  if (!is.data.table(importance_matrix))  {
    stop("importance_matrix: must be a data.table")
  }

  imp_names <- colnames(importance_matrix)
  if (is.null(measure)) {
    if (all(c("Feature", "Gain") %in% imp_names)) {
      measure <- "Gain"
    } else if (all(c("Feature", "Weight") %in% imp_names)) {
      measure <- "Weight"
    } else {
      stop("Importance matrix column names are not as expected!")
    }
  } else {
    if (!measure %in% imp_names)
      stop("Invalid `measure`")
    if (!"Feature" %in% imp_names)
      stop("Importance matrix column names are not as expected!")
  }

  # also aggregate, just in case when the values were not yet summed up by feature
  importance_matrix <- importance_matrix[, Importance := sum(get(measure)), by = Feature]

  # make sure it's ordered
  importance_matrix <- importance_matrix[order(-abs(Importance))]

  if (!is.null(top_n)) {
    top_n <- min(top_n, nrow(importance_matrix))
    importance_matrix <- head(importance_matrix, top_n)
  }
  if (rel_to_first) {
    importance_matrix[, Importance := Importance / max(abs(Importance))]
  }
  if (is.null(cex)) {
    cex <- 2.5 / log2(1 + nrow(importance_matrix))
  }

  if (plot) {
    original_mar <- par()$mar

    # reset margins so this function doesn't have side effects
    on.exit({
        par(mar = original_mar)
    })

    mar <- original_mar
    if (!is.null(left_margin))
      mar[2] <- left_margin
    par(mar = mar)

    # reverse the order of rows to have the highest ranked at the top
    importance_matrix[rev(seq_len(nrow(importance_matrix))),
                      barplot(Importance, horiz = TRUE, border = NA, cex.names = cex,
                              names.arg = Feature, las = 1, ...)]
  }

  invisible(importance_matrix)
}

# Avoid error messages during CRAN check.
# The reason is that these variables are never declared
# They are mainly column names inferred by Data.table...
globalVariables(c("Feature", "Importance"))
