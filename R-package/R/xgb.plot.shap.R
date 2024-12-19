#' SHAP dependence plots
#'
#' Visualizes SHAP values against feature values to gain an impression of feature effects.
#'
#' @param data The data to explain as a `matrix`, `dgCMatrix`, or `data.frame`.
#' @param shap_contrib Matrix of SHAP contributions of `data`.
#'   The default (`NULL`) computes it from `model` and `data`.
#' @param features Vector of column indices or feature names to plot. When `NULL`
#'   (default), the `top_n` most important features are selected by [xgb.importance()].
#' @param top_n How many of the most important features (<= 100) should be selected?
#'   By default 1 for SHAP dependence and 10 for SHAP summary.
#'   Only used when `features = NULL`.
#' @param model An `xgb.Booster` model. Only required when `shap_contrib = NULL` or
#'   `features = NULL`.
#' @param trees Passed to [xgb.importance()] when `features = NULL`.
#' @param target_class Only relevant for multiclass models. The default (`NULL`)
#'   averages the SHAP values over all classes. Pass a (0-based) class index
#'   to show only SHAP values of that class.
#' @param approxcontrib Passed to [predict.xgb.Booster()] when `shap_contrib = NULL`.
#' @param subsample Fraction of data points randomly picked for plotting.
#'   The default (`NULL`) will use up to 100k data points.
#' @param n_col Number of columns in a grid of plots.
#' @param col Color of the scatterplot markers.
#' @param pch Scatterplot marker.
#' @param discrete_n_uniq Maximal number of unique feature values to consider the
#'   feature as discrete.
#' @param discrete_jitter Jitter amount added to the values of discrete features.
#' @param ylab The y-axis label in 1D plots.
#' @param plot_NA Should contributions of cases with missing values be plotted?
#'   Default is `TRUE`.
#' @param col_NA Color of marker for missing value contributions.
#' @param pch_NA Marker type for `NA` values.
#' @param pos_NA Relative position of the x-location where `NA` values are shown:
#'   `min(x) + (max(x) - min(x)) * pos_NA`.
#' @param plot_loess Should loess-smoothed curves be plotted? (Default is `TRUE`).
#'   The smoothing is only done for features with more than 5 distinct values.
#' @param col_loess Color of loess curves.
#' @param span_loess The `span` parameter of [stats::loess()].
#' @param which Whether to do univariate or bivariate plotting. Currently, only "1d" is implemented.
#' @param plot Should the plot be drawn? (Default is `TRUE`).
#'   If `FALSE`, only a list of matrices is returned.
#' @param ... Other parameters passed to [graphics::plot()].
#'
#' @details
#'
#' These scatterplots represent how SHAP feature contributions depend of feature values.
#' The similarity to partial dependence plots is that they also give an idea for how feature values
#' affect predictions. However, in partial dependence plots, we see marginal dependencies
#' of model prediction on feature value, while SHAP dependence plots display the estimated
#' contributions of a feature to the prediction for each individual case.
#'
#' When `plot_loess = TRUE`, feature values are rounded to three significant digits and
#' weighted LOESS is computed and plotted, where the weights are the numbers of data points
#' at each rounded value.
#'
#' Note: SHAP contributions are on the scale of the model margin.
#' E.g., for a logistic binomial objective, the margin is on log-odds scale.
#' Also, since SHAP stands for "SHapley Additive exPlanation" (model prediction = sum of SHAP
#' contributions for all features + bias), depending on the objective used, transforming SHAP
#' contributions for a feature from the marginal to the prediction space is not necessarily
#' a meaningful thing to do.
#'
#' @return
#' In addition to producing plots (when `plot = TRUE`), it silently returns a list of two matrices:
#' - `data`: Feature value matrix.
#' - `shap_contrib`: Corresponding SHAP value matrix.
#'
#' @references
#' 1. Scott M. Lundberg, Su-In Lee, "A Unified Approach to Interpreting Model Predictions",
#'    NIPS Proceedings 2017, <https://arxiv.org/abs/1705.07874>
#' 2. Scott M. Lundberg, Su-In Lee, "Consistent feature attribution for tree ensembles",
#'    <https://arxiv.org/abs/1706.06060>
#'
#' @examples
#'
#' data(agaricus.train, package = "xgboost")
#' data(agaricus.test, package = "xgboost")
#'
#' ## Keep the number of threads to 1 for examples
#' nthread <- 1
#' data.table::setDTthreads(nthread)
#' nrounds <- 20
#'
#' model_binary <- xgboost(
#'   agaricus.train$data, factor(agaricus.train$label),
#'   nrounds = nrounds,
#'   verbosity = 0L,
#'   learning_rate = 0.1,
#'   max_depth = 3L,
#'   subsample = 0.5,
#'   nthreads = nthread
#' )
#'
#' xgb.plot.shap(agaricus.test$data, model = model_binary, features = "odor=none")
#'
#' contr <- predict(model_binary, agaricus.test$data, type = "contrib")
#' xgb.plot.shap(agaricus.test$data, contr, model = model_binary, top_n = 12, n_col = 3)
#'
#' # Summary plot
#' xgb.ggplot.shap.summary(agaricus.test$data, contr, model = model_binary, top_n = 12)
#'
#' # Multiclass example - plots for each class separately:
#' x <- as.matrix(iris[, -5])
#' set.seed(123)
#' is.na(x[sample(nrow(x) * 4, 30)]) <- TRUE # introduce some missing values
#'
#' model_multiclass <- xgboost(
#'   x, iris$Species,
#'   nrounds = nrounds,
#'   verbosity = 0,
#'   max_depth = 2,
#'   subsample = 0.5,
#'   nthreads = nthread
#' )
#' nclass <- 3
#' trees0 <- seq(from = 1, by = nclass, length.out = nrounds)
#' col <- rgb(0, 0, 1, 0.5)
#'
#' xgb.plot.shap(
#'   x,
#'   model = model_multiclass,
#'   trees = trees0,
#'   target_class = 0,
#'   top_n = 4,
#'   n_col = 2,
#'   col = col,
#'   pch = 16,
#'   pch_NA = 17
#' )
#'
#' xgb.plot.shap(
#'   x,
#'   model = model_multiclass,
#'   trees = trees0 + 1,
#'   target_class = 1,
#'   top_n = 4,
#'   n_col = 2,
#'   col = col,
#'   pch = 16,
#'   pch_NA = 17
#' )
#'
#' xgb.plot.shap(
#'   x,
#'   model = model_multiclass,
#'   trees = trees0 + 2,
#'   target_class = 2,
#'   top_n = 4,
#'   n_col = 2,
#'   col = col,
#'   pch = 16,
#'   pch_NA = 17
#' )
#'
#' # Summary plot
#' xgb.ggplot.shap.summary(x, model = model_multiclass, target_class = 0, top_n = 4)
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
    max_observations = 100000
  )
  data <- data_list[["data"]]
  shap_contrib <- data_list[["shap_contrib"]]
  features <- colnames(data)

  which <- match.arg(which)
  if (which == "2d")
    stop("2D plots are not implemented yet")

  if (n_col > length(features)) n_col <- length(features)
  if (plot && which == "1d") {
    op <- par(mfrow = c(ceiling(length(features) / n_col), n_col),
              oma = c(0, 0, 0, 0) + 0.2,
              mar = c(3.5, 3.5, 0, 0) + 0.1,
              mgp = c(1.7, 0.6, 0))
    for (f in features) {
      ord <- order(data[, f])
      x <- data[, f][ord]
      y <- shap_contrib[, f][ord]
      x_lim <- range(x, na.rm = TRUE)
      y_lim <- range(y, na.rm = TRUE)
      do_na <- plot_NA && anyNA(x)
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
        # compress x to 3 digits, and mean-aggregate y
        zz <- data.table(x = signif(x, 3), y)[, .(.N, y = mean(y)), x]
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

#' SHAP summary plot
#'
#' Visualizes SHAP contributions of different features.
#'
#' A point plot (each point representing one observation from `data`) is
#' produced for each feature, with the points plotted on the SHAP value axis.
#' Each point (observation) is coloured based on its feature value.
#'
#' The plot allows to see which features have a negative / positive contribution
#' on the model prediction, and whether the contribution is different for larger
#' or smaller values of the feature. Inspired by the summary plot of
#' <https://github.com/shap/shap>.
#'
#' @inheritParams xgb.plot.shap
#'
#' @return A `ggplot2` object.
#' @export
#'
#' @examples
#' # See examples in xgb.plot.shap()
#'
#' @seealso [xgb.plot.shap()], [xgb.ggplot.shap.summary()],
#'   and the Python library <https://github.com/shap/shap>.
xgb.plot.shap.summary <- function(data, shap_contrib = NULL, features = NULL, top_n = 10, model = NULL,
                                  trees = NULL, target_class = NULL, approxcontrib = FALSE, subsample = NULL) {
  # Only ggplot implementation is available.
  xgb.ggplot.shap.summary(data, shap_contrib, features, top_n, model, trees, target_class, approxcontrib, subsample)
}

#' Prepare data for SHAP plots
#'
#' Internal function used in [xgb.plot.shap()], [xgb.plot.shap.summary()], etc.
#'
#' @inheritParams xgb.plot.shap
#' @param max_observations Maximum number of observations to consider.
#' @keywords internal
#' @noRd
#'
#' @return
#' A list containing:
#' - `data`: The matrix of feature values.
#' - `shap_contrib`: The matrix with corresponding SHAP values.
xgb.shap.data <- function(data, shap_contrib = NULL, features = NULL, top_n = 1, model = NULL,
                          trees = NULL, target_class = NULL, approxcontrib = FALSE,
                          subsample = NULL, max_observations = 100000) {
  if (!inherits(data, c("matrix", "dsparseMatrix", "data.frame")))
    stop("data: must be matrix, sparse matrix, or data.frame.")
  if (inherits(data, "data.frame") && length(class(data)) > 1L) {
    data <- as.data.frame(data)
  }

  if (is.null(shap_contrib) && (is.null(model) || !inherits(model, "xgb.Booster")))
    stop("when shap_contrib is not provided, one must provide an xgb.Booster model")

  if (is.null(features) && (is.null(model) || !inherits(model, "xgb.Booster")))
    stop("when features are not provided, one must provide an xgb.Booster model to rank the features")

  last_dim <- function(v) dim(v)[length(dim(v))]

  if (!is.null(shap_contrib) &&
      (!is.array(shap_contrib) || nrow(shap_contrib) != nrow(data) || last_dim(shap_contrib) != ncol(data) + 1))
    stop("shap_contrib is not compatible with the provided data")

  if (is.character(features) && is.null(colnames(data)))
    stop("either provide `data` with column names or provide `features` as column indices")

  model_feature_names <- NULL
  if (is.null(features) && !is.null(model)) {
    model_feature_names <- xgb.feature_names(model)
  }
  if (is.null(model_feature_names) && xgb.num_feature(model) != ncol(data))
    stop("if model has no feature_names, columns in `data` must match features in model")

  if (!is.null(subsample)) {
    if (subsample <= 0 || subsample >= 1) {
      stop("'subsample' must be a number between zero and one (non-inclusive).")
    }
    sample_size <- as.integer(subsample * nrow(data))
    if (sample_size < 2) {
      stop("Sampling fraction involves less than 2 rows.")
    }
    idx <- sample(x = seq_len(nrow(data)), size = sample_size, replace = FALSE)
  } else {
    idx <- seq_len(min(nrow(data), max_observations))
  }
  data <- data[idx, ]
  if (is.null(colnames(data))) {
    colnames(data) <- paste0("X", seq_len(ncol(data)))
  }

  reshape_3d_shap_contrib <- function(shap_contrib, target_class) {
    # multiclass: either choose a class or merge
    if (is.list(shap_contrib)) {
      if (!is.null(target_class)) {
        shap_contrib <- shap_contrib[[target_class + 1]]
      } else {
        shap_contrib <- Reduce("+", lapply(shap_contrib, abs))
      }
    } else if (length(dim(shap_contrib)) > 2) {
      if (!is.null(target_class)) {
        orig_shape <- dim(shap_contrib)
        shap_contrib <- shap_contrib[, target_class + 1, , drop = TRUE]
        if (!is.matrix(shap_contrib)) {
          shap_contrib <- matrix(shap_contrib, orig_shape[c(1L, 3L)])
        }
      } else {
        shap_contrib <- apply(abs(shap_contrib), c(1L, 3L), sum)
      }
    }
    return(shap_contrib)
  }

  if (is.null(shap_contrib)) {
    shap_contrib <- predict.xgb.Booster(
      model,
      newdata = data,
      predcontrib = TRUE,
      approxcontrib = approxcontrib
    )
  }
  shap_contrib <- reshape_3d_shap_contrib(shap_contrib, target_class)
  if (is.null(colnames(shap_contrib))) {
    colnames(shap_contrib) <- paste0("X", seq_len(ncol(data)))
  }

  if (is.null(features)) {
    if (!is.null(model_feature_names)) {
      imp <- xgb.importance(model = model, trees = trees)
    } else {
      imp <- xgb.importance(model = model, trees = trees, feature_names = colnames(data))
    }
    top_n <- top_n[1]
    if (top_n < 1 || top_n > 100) stop("top_n: must be an integer within [1, 100]")
    features <- imp$Feature[seq_len(min(top_n, NROW(imp)))]
  }
  if (is.character(features)) {
    features <- match(features, colnames(data))
  }

  shap_contrib <- shap_contrib[, features, drop = FALSE]
  data <- data[, features, drop = FALSE]

  list(
    data = data,
    shap_contrib = shap_contrib
  )
}
