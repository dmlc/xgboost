#' @export
dim.xgb.DMatrix <- function(x) {
  if ("dimensions" %in% names(attributes(x))) {
    return(attr(x, "dimensions"))
  }
  NULL
}
