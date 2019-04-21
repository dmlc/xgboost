#' SHAP contribution dependency plots
#'
#' Visualizing the SHAP feature contribution to prediction dependencies on feature value.
#'
#' @param data data as a \code{matrix} or \code{dgCMatrix}.
#' @param shap_contrib a matrix of SHAP contributions that was computed earlier for the above
#'          \code{data}. When it is NULL, it is computed internally using \code{model} and \code{data}.
#' @param features a vector of either column indices or of feature names to plot. When it is NULL,
#'          feature importance is calculated, and \code{top_n} high ranked features are taken.
#' @param top_n when \code{features} is NULL, top_n [1, 100] most important features in a model are taken.
#' @param model an \code{xgb.Booster} model. It has to be provided when either \code{shap_contrib}
#'          or \code{features} is missing.
#' @param trees passed to \code{\link{xgb.importance}} when \code{features = NULL}.
#' @param target_class is only relevant for multiclass models. When it is set to a 0-based class index,
#'          only SHAP contributions for that specific class are used.
#'          If it is not set, SHAP importances are averaged over all classes.
#' @param approxcontrib passed to \code{\link{predict.xgb.Booster}} when \code{shap_contrib = NULL}.
#' @param subsample a random fraction of data points to use for plotting. When it is NULL,
#'          it is set so that up to 100K data points are used.
#' @param n_col a number of columns in a grid of plots.
#' @param col color of the scatterplot markers.
#' @param pch scatterplot marker.
#' @param discrete_n_uniq a maximal number of unique values in a feature to consider it as discrete.
#' @param discrete_jitter an \code{amount} parameter of jitter added to discrete features' positions.
#' @param ylab a y-axis label in 1D plots.
#' @param plot_NA whether the contributions of cases with missing values should also be plotted.
#' @param col_NA a color of marker for missing value contributions.
#' @param pch_NA a marker type for NA values.
#' @param pos_NA a relative position of the x-location where NA values are shown:
#'          \code{min(x) + (max(x) - min(x)) * pos_NA}.
#' @param plot_loess whether to plot loess-smoothed curves. The smoothing is only done for features with
#'          more than 5 distinct values.
#' @param col_loess a color to use for the loess curves.
#' @param span_loess the \code{span} parameter in \code{\link[stats]{loess}}'s call.
#' @param which whether to do univariate or bivariate plotting. NOTE: only 1D is implemented so far.
#' @param plot whether a plot should be drawn. If FALSE, only a lits of matrices is returned.
#' @param ... other parameters passed to \code{plot}.
#'
#' @details
#'
#' These scatterplots represent how SHAP feature contributions depend of feature values.
#' The similarity to partial dependency plots is that they also give an idea for how feature values
#' affect predictions. However, in partial dependency plots, we usually see marginal dependencies
#' of model prediction on feature value, while SHAP contribution dependency plots display the estimated
#' contributions of a feature to model prediction for each individual case.
#'
#' When \code{plot_loess = TRUE} is set, feature values are rounded to 3 significant digits and
#' weighted LOESS is computed and plotted, where weights are the numbers of data points
#' at each rounded value.
#'
#' Note: SHAP contributions are shown on the scale of model margin. E.g., for a logistic binomial objective,
#' the margin is prediction before a sigmoidal transform into probability-like values.
#' Also, since SHAP stands for "SHapley Additive exPlanation" (model prediction = sum of SHAP
#' contributions for all features + bias), depending on the objective used, transforming SHAP
#' contributions for a feature from the marginal to the prediction space is not necessarily
#' a meaningful thing to do.
#'
#' @return
#'
#' In addition to producing plots (when \code{plot=TRUE}), it silently returns a list of two matrices:
#' \itemize{
#'  \item \code{data} the values of selected features;
#'  \item \code{shap_contrib} the contributions of selected features.
#' }
#'
#' @references
#'
#' Scott M. Lundberg, Su-In Lee, "A Unified Approach to Interpreting Model Predictions", NIPS Proceedings 2017, \url{https://arxiv.org/abs/1705.07874}
#'
#' Scott M. Lundberg, Su-In Lee, "Consistent feature attribution for tree ensembles", \url{https://arxiv.org/abs/1706.06060}
#'
#' @examples
#'
#' data(agaricus.train, package='xgboost')
#' data(agaricus.test, package='xgboost')
#'
#' bst <- xgboost(agaricus.train$data, agaricus.train$label, nrounds = 50,
#'                eta = 0.1, max_depth = 3, subsample = .5,
#'                method = "hist", objective = "binary:logistic", nthread = 2, verbose = 0)
#'
#' xgb.plot.shap(agaricus.test$data, model = bst, features = "odor=none")
#' contr <- predict(bst, agaricus.test$data, predcontrib = TRUE)
#' xgb.plot.shap(agaricus.test$data, contr, model = bst, top_n = 12, n_col = 3)
#'
#' # multiclass example - plots for each class separately:
#' nclass <- 3
#' nrounds <- 20
#' x <- as.matrix(iris[, -5])
#' set.seed(123)
#' is.na(x[sample(nrow(x) * 4, 30)]) <- TRUE # introduce some missing values
#' mbst <- xgboost(data = x, label = as.numeric(iris$Species) - 1, nrounds = nrounds,
#'                 max_depth = 2, eta = 0.3, subsample = .5, nthread = 2,
#'                 objective = "multi:softprob", num_class = nclass, verbose = 0)
#' trees0 <- seq(from=0, by=nclass, length.out=nrounds)
#' col <- rgb(0, 0, 1, 0.5)
#' xgb.plot.shap(x, model = mbst, trees = trees0, target_class = 0, top_n = 4,
#'               n_col = 2, col = col, pch = 16, pch_NA = 17)
#' xgb.plot.shap(x, model = mbst, trees = trees0 + 1, target_class = 1, top_n = 4,
#'               n_col = 2, col = col, pch = 16, pch_NA = 17)
#' xgb.plot.shap(x, model = mbst, trees = trees0 + 2, target_class = 2, top_n = 4,
#'               n_col = 2, col = col, pch = 16, pch_NA = 17)
#'
#' @rdname xgb.plot.shap
#' @export
xgb.plot.shap <- function(data, shap_contrib = NULL, features = NULL, top_n = 1, model = NULL,
                          trees = NULL, target_class = NULL, approxcontrib = FALSE,
                          subsample = NULL, n_col = 1, col = rgb(0, 0, 1, 0.2), pch = '.',
                          discrete_n_uniq = 5, discrete_jitter = 0.01, ylab = "SHAP",
                          plot_NA = TRUE, col_NA = rgb(0.7, 0, 1, 0.6), pch_NA = '.', pos_NA = 1.07,
                          plot_loess = TRUE, col_loess = 2, span_loess = 0.5,
                          which = c("1d", "2d"), plot = TRUE, ...) {

  if (!is.matrix(data) && !inherits(data, "dgCMatrix"))
    stop("data: must be either matrix or dgCMatrix")

  if (is.null(shap_contrib) && (is.null(model) || !inherits(model, "xgb.Booster")))
    stop("when shap_contrib is not provided, one must provide an xgb.Booster model")

  if (is.null(features) && (is.null(model) || !inherits(model, "xgb.Booster")))
    stop("when features are not provided, one must provide an xgb.Booster model to rank the features")

  if (!is.null(shap_contrib) &&
      (!is.matrix(shap_contrib) || nrow(shap_contrib) != nrow(data) || ncol(shap_contrib) != ncol(data) + 1))
    stop("shap_contrib is not compatible with the provided data")

  nsample <- if (is.null(subsample)) min(100000, nrow(data)) else as.integer(subsample * nrow(data))
  idx <- sample(1:nrow(data), nsample)
  data <- data[idx,]

  if (is.null(shap_contrib)) {
    shap_contrib <- predict(model, data, predcontrib = TRUE, approxcontrib = approxcontrib)
  } else {
    shap_contrib <- shap_contrib[idx,]
  }

  which <- match.arg(which)
  if (which == "2d")
    stop("2D plots are not implemented yet")

  if (is.null(features)) {
    imp <- xgb.importance(model = model, trees = trees)
    top_n <- as.integer(top_n[1])
    if (top_n < 1 && top_n > 100)
      stop("top_n: must be an integer within [1, 100]")
    features <- imp$Feature[1:min(top_n, NROW(imp))]
  }

  if (is.character(features)) {
    if (is.null(colnames(data)))
      stop("Either provide `data` with column names or provide `features` as column indices")
    features <- match(features, colnames(data))
  }

  if (n_col > length(features)) n_col <- length(features)

  if (is.list(shap_contrib)) { # multiclass: either choose a class or merge
    shap_contrib <- if (!is.null(target_class)) shap_contrib[[target_class + 1]]
                    else Reduce("+", lapply(shap_contrib, abs))
  }

  shap_contrib <- shap_contrib[, features, drop = FALSE]
  data <- data[, features, drop = FALSE]
  cols <- colnames(data)
  if (is.null(cols)) cols <- colnames(shap_contrib)
  if (is.null(cols)) cols <- paste0('X', 1:ncol(data))
  colnames(data) <- cols
  colnames(shap_contrib) <- cols

  if (plot && which == "1d") {
    op <- par(mfrow = c(ceiling(length(features) / n_col), n_col),
              oma = c(0,0,0,0) + 0.2,
              mar = c(3.5,3.5,0,0) + 0.1,
              mgp = c(1.7, 0.6, 0))
    for (f in cols) {
      ord <- order(data[, f])
      x <- data[, f][ord]
      y <- shap_contrib[, f][ord]
      x_lim <- range(x, na.rm = TRUE)
      y_lim <- range(y, na.rm = TRUE)
      do_na <- plot_NA && any(is.na(x))
      if (do_na) {
        x_range <- diff(x_lim)
        loc_na <- min(x, na.rm = TRUE) + x_range * pos_NA
        x_lim <- range(c(x_lim, loc_na))
      }
      x_uniq <- unique(x)
      x2plot <- x
      # add small jitter for discrete features with <= 5 distinct values
      if (length(x_uniq) <= discrete_n_uniq)
        x2plot <- jitter(x, amount = discrete_jitter * min(diff(x_uniq), na.rm = TRUE))
      plot(x2plot, y, pch = pch, xlab = f, col = col, xlim = x_lim, ylim = y_lim, ylab = ylab, ...)
      grid()
      if (plot_loess) {
        # compress x to 3 digits, and mean-aggredate y
        zz <- data.table(x = signif(x, 3), y)[, .(.N, y=mean(y)), x]
        if (nrow(zz) <= 5) {
          lines(zz$x, zz$y, col = col_loess)
        } else {
          lo <- stats::loess(y ~ x, data = zz, weights = zz$N, span = span_loess)
          zz$y_lo <- predict(lo, zz, type = "link")
          lines(zz$x, zz$y_lo, col = col_loess)
        }
      }
      if (do_na) {
        i_na <- which(is.na(x))
        x_na <- rep(loc_na, length(i_na))
        x_na <- jitter(x_na, amount = x_range * 0.01)
        points(x_na, y[i_na], pch = pch_NA, col = col_NA)
      }
    }
    par(op)
  }
  if (plot && which == "2d") {
    # TODO
    warning("Bivariate plotting is currently not available.")
  }
  invisible(list(data = data, shap_contrib = shap_contrib))
}
