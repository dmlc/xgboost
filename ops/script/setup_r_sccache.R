# Configure R package builds to use sccache.
#
# Traditional R builds use wrapped compilers with autotools, while XGBoost's CMake build
# uses the original compilers with CMake's compiler launcher variables.

makevars <- file.path(path.expand("~"), ".R", "Makevars")
if (file.exists(makevars)) {
  stop(
    makevars,
    " already exists; aborting to avoid overwriting it.",
    call. = FALSE
  )
}

dir.create(dirname(makevars), recursive = TRUE, showWarnings = FALSE)
# `:=` captures the original compilers before the later sccache assignments. Assign CMake
# compiler first, then re-assign CC.
writeLines(
  c(
    "CMAKE_C_COMPILER := $(CC)",
    "CMAKE_CXX_COMPILER := $(CXX)",
    "CC := sccache $(CC)",
    "CXX := sccache $(CXX)",
    "CXX11 := sccache $(CXX11)",
    "CXX14 := sccache $(CXX14)",
    "CXX17 := sccache $(CXX17)",
    "CXX20 := sccache $(CXX20)",
    "CMAKE_C_COMPILER_LAUNCHER = sccache",
    "CMAKE_CXX_COMPILER_LAUNCHER = sccache",
    "CMAKE_CUDA_COMPILER_LAUNCHER = sccache"
  ),
  makevars
)
message("Configured the XGBoost R package to use sccache via ", makevars)
