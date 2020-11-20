#' @export
xgb.set.config <- function(...) {
  new_config <- list(...)
  .Call(XGBSetGlobalConfig_R, jsonlite::toJSON(new_config, auto_unbox = TRUE))
  return(invisible(NULL))
}

#' @export
xgb.get.config <- function() {
  config <- .Call(XGBGetGlobalConfig_R)
  return(jsonlite::fromJSON(config))
}
