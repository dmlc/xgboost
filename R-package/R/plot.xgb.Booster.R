#------------------------------
# helper functions for plotting 
#

plot.xgb.Booster <- function(object, ntrees, index, is_factor, data, ...) {
  if (missing(data)) stop("Data is not saved in xgboost model object, so you have to provide that...")
  if (is_factor) {
    pred_var_x <- levels(factor(data[, index], levels = unique(data[, index]), exclude = NULL))
    pred_var_y <- numeric(length(pred_var_x))
    for (i in seq_along(pred_var_x)) {
      if (is.na(pred_var_x[i])) { 
        indices_bucket <- which(is.na(data[, index]))
      } else {
        indices_bucket <- which(data[, index] == pred_var_x[i])
      }
      if (length(indices_bucket) == 0) {
        stop("Something is wrong with this re-implementation...")
      }
      preds_bucket <- xgboost::predict(object, data[indices_bucket, , drop = FALSE],
        missing = NA, ntreelimit = ntrees)
      pred_var_y[i] <- median(preds_bucket, na.rm = TRUE)
    }
  } else {
    .length <- 100
    pred_var_x <- seq(min(data[, index], na.rm = TRUE), 
      max(data[, index], na.rm = TRUE), 
      length = .length)
    pred_var_y <- numeric(.length - 1)
    for (i in 2:.length) {
      indices_bucket <- which(data[, index] >= pred_var_x[i-1] & data[, index] < pred_var_x[i])
      # Use previous interpolation
      if (length(indices_bucket) == 0) { 
        pred_var_y[i - 1] <- pred_var_y[i - 2]
      } else {
        preds_bucket <- xgboost::predict(object, data[indices_bucket, , drop = FALSE], 
          missing = NA, ntreelimit = ntrees)
        pred_var_y[i - 1] <- median(preds_bucket, na.rm = TRUE)
      }
    }
    pred_var_x <- pred_var_x[-1]
  }
  plot(pred_var_x, pred_var_y, ...)
}
