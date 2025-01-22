#' Plot feature importance
#'
#' Represents previously calculated feature importance as a bar graph.
#' - `xgb.plot.importance()` uses base R graphics, while
#' - `xgb.ggplot.importance()` uses "ggplot".
#'
#' @details
#' The graph represents each feature as a horizontal bar of length proportional to the
#' importance of a feature. Features are sorted by decreasing importance.
#' It works for both "gblinear" and "gbtree" models.
#'
#' When `rel_to_first = FALSE`, the values would be plotted as in `importance_matrix`.
#' For a "gbtree" model, that would mean being normalized to the total of 1
#' ("what is feature's importance contribution relative to the whole model?").
#' For linear models, `rel_to_first = FALSE` would show actual values of the coefficients.
#' Setting `rel_to_first = TRUE` allows to see the picture from the perspective of
#' "what is feature's importance contribution relative to the most important feature?"
#'
#' The "ggplot" backend performs 1-D clustering of the importance values,
#' with bar colors corresponding to different clusters having similar importance values.
#'
#' @param importance_matrix A `data.table` as returned by [xgb.importance()].
#' @param top_n Maximal number of top features to include into the plot.
#' @param measure The name of importance measure to plot.
#'   When `NULL`, 'Gain' would be used for trees and 'Weight' would be used for gblinear.
#' @param rel_to_first Whether importance values should be represented as relative to
#'   the highest ranked feature, see Details.
#' @param left_margin Adjust the left margin size to fit feature names.
#'   When `NULL`, the existing `par("mar")` is used.
#' @param cex Passed as `cex.names` parameter to [graphics::barplot()].
#' @param plot Should the barplot be shown? Default is `TRUE`.
#' @param n_clusters A numeric vector containing the min and the max range
#'   of the possible number of clusters of bars.
#' @param ... Other parameters passed to [graphics::barplot()]
#'   (except `horiz`, `border`, `cex.names`, `names.arg`, and `las`).
#'   Only used in `xgb.plot.importance()`.
#' @return
#' The return value depends on the function:
#' - `xgb.plot.importance()`: Invisibly, a "data.table" with `n_top` features sorted
#'   by importance. If `plot = TRUE`, the values are also plotted as barplot.
#' - `xgb.ggplot.importance()`: A customizable "ggplot" object.
#'   E.g., to change the title, set `+ ggtitle("A GRAPH NAME")`.
#'
#' @seealso [graphics::barplot()]
#'
#' @examples
#' data(agaricus.train)
#'
#' ## Keep the number of threads to 2 for examples
#' nthread <- 2
#' data.table::setDTthreads(nthread)
#'
#' model <- xgboost(
#'   agaricus.train$data, factor(agaricus.train$label),
#'   nrounds = 2,
#'   max_depth = 3,
#'   nthreads = nthread
#' )
#'
#' importance_matrix <- xgb.importance(model)
#' xgb.plot.importance(
#'   importance_matrix, rel_to_first = TRUE, xlab = "Relative importance"
#' )
#'
#' gg <- xgb.ggplot.importance(
#'   importance_matrix, measure = "Frequency", rel_to_first = TRUE
#' )
#' gg
#' gg + ggplot2::ylab("Frequency")
#'
#' @rdname xgb.plot.importance
#' @export
xgb.plot.importance <- function(importance_matrix = NULL, top_n = NULL, measure = NULL,
                                rel_to_first = FALSE, left_margin = 10, cex = NULL, plot = TRUE, ...) {
  check.deprecation(deprecated_plotimp_params, match.call(), ..., allow_unrecognized = TRUE)
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
  importance_matrix <- importance_matrix[
    , lapply(.SD, sum)
    , .SDcols = setdiff(names(importance_matrix), "Feature")
    , by = Feature
  ][
    , Importance := get(measure)
  ]

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
