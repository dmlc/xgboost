#' Set and get global configuration
#'
#' Global configuration consists of a collection of parameters that can be applied in the global
#' scope. See \url{https://xgboost.readthedocs.io/en/stable/parameter.html} for the full list of
#' parameters supported in the global configuration. Use `xgb.set.config()` to update the
#' values of one or more global-scope parameters. Use `xgb.get.config()` to fetch the current
#' values of all global-scope parameters (listed in
#' \url{https://xgboost.readthedocs.io/en/stable/parameter.html}).
#'
#' @details
#' Note that serialization-related functions might use a globally-configured number of threads,
#' which is managed by the system's OpenMP (OMP) configuration instead. Typically, XGBoost methods
#' accept an `nthreads` parameter, but some methods like [readRDS()] might get executed before such
#' parameter can be supplied.
#'
#' The number of OMP threads can in turn be configured for example through an environment variable
#' `OMP_NUM_THREADS` (needs to be set before R is started), or through `RhpcBLASctl::omp_set_num_threads`.
#' @rdname xgbConfig
#' @name xgb.set.config, xgb.get.config
#' @export xgb.set.config xgb.get.config
#' @param ... List of parameters to be set, as keyword arguments
#' @return
#' `xgb.set.config()` returns `TRUE` to signal success. `xgb.get.config()` returns
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
