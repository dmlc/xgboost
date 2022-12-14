## Install dependencies of R package for testing. The list might not be
## up-to-date, check DESCRIPTION for the latest list and update this one if
## inconsistent is found.
pkgs <- c(
  ## CI
  "caret",
  "pkgbuild",
  "roxygen2",
  "XML",
  "cplm",
  "e1071",
  ## suggests
  "knitr",
  "rmarkdown",
  "ggplot2",
  "DiagrammeR",
  "Ckmeans.1d.dp",
  "vcd",
  "lintr",
  "testthat",
  "igraph",
  "float",
  "titanic",
  ## imports
  "Matrix",
  "methods",
  "data.table",
  "jsonlite"
)

ncpus <- parallel::detectCores()
print(paste0("Using ", ncpus, " cores to install dependencies."))

if (.Platform$OS.type == "unix") {
  print("Installing source packages on unix.")
  install.packages(
    pkgs,
    repo = "https://cloud.r-project.org",
    dependencies = c("Depends", "Imports", "LinkingTo"),
    Ncpus = parallel::detectCores()
  )
} else {
  print("Installing binary packages on Windows.")
  install.packages(
    pkgs,
    repo = "https://cloud.r-project.org",
    dependencies = c("Depends", "Imports", "LinkingTo"),
    Ncpus = parallel::detectCores(),
    type = "binary"
  )
}
