#!/usr/bin/env Rscript
# Configure the XGBoost R package's CMake build to use sccache.
# This script creates R's user Makevars file with CMake compiler launchers.

makevars <- file.path(path.expand("~"), ".R", "Makevars")
if (file.exists(makevars)) {
  stop(
    makevars,
    " already exists; aborting to avoid overwriting it.",
    call. = FALSE
  )
}

dir.create(dirname(makevars), recursive = TRUE, showWarnings = FALSE)
writeLines(
  c(
    "XGBOOST_CMAKE_C_COMPILER_LAUNCHER = sccache",
    "XGBOOST_CMAKE_CXX_COMPILER_LAUNCHER = sccache",
    "XGBOOST_CMAKE_CUDA_COMPILER_LAUNCHER = sccache"
  ),
  makevars
)
message("Configured the XGBoost R package to use sccache via ", makevars)
