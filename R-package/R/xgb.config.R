#' Global configuration consists of a collection of parameters that can be applied in the global
#' scope. See \url{https://xgboost.readthedocs.io/en/stable/parameter.html} for the full list of
#' parameters supported in the global configuration. Use \code{xgb.set.config} to update the
#' values of one or more global-scope parameters. Use \code{xgb.get.config} to fetch the current
#' values of all global-scope parameters (listed in
#' \url{https://xgboost.readthedocs.io/en/stable/parameter.html}).
#'
#' @rdname xgbConfig
#' @title Set and get global configuration
#' @name xgb.set.config, xgb.get.config
#' @export xgb.set.config xgb.get.config
#' @param ... List of parameters to be set, as keyword arguments
#' @return
#' \code{xgb.set.config} returns \code{TRUE} to signal success. \code{xgb.get.config} returns
#' a list containing all global-scope parameters and their values.
#'
#' @examples
#' # Set verbosity level to silent (0)
#' xgb.set.config(verbosity = 0)
#' # Now global verbosity level is 0
#' config <- xgb.get.config()
#' print(config$verbosity)
#' # Set verbosity level to warning (1)
#' xgb.set.config(verbosity = 1)
#' # Now global verbosity level is 1
#' config <- xgb.get.config()
#' print(config$verbosity)
xgb.set.config <- function(...) {
  new_config <- list(...)
  .Call(XGBSetGlobalConfig_R, jsonlite::toJSON(new_config, auto_unbox = TRUE))
  return(TRUE)
}

#' @rdname xgbConfig
xgb.get.config <- function() {
  config <- .Call(XGBGetGlobalConfig_R)
  return(jsonlite::fromJSON(config))
}
