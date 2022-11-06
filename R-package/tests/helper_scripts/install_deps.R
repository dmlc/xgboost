## Install dependencies of R package for testing. The list might not be
## up-to-date, check DESCRIPTION for the latest list and update this one if
## inconsistent is found.
pkgs <- c(
  "XML",
  "igraph",
  "data.table",
  "ggplot2",
  "DiagrammeR",
  "Ckmeans.1d.dp",
  "vcd",
  "testthat",
  "lintr",
  "knitr",
  "rmarkdown",
  "e1071",
  "cplm",
  "devtools",
  "float",
  "titanic",
)

ncpus <- parallel::detectCores()
print(paste0("Using ", ncpus, " cores to install dependencies."))

if (.Platform$OS.type == "unix") {
  print("Installing source package on unix.")
  install.packages(
    pkgs,
    repo = "http://cloud.r-project.org",
    dependencies = c("Depends", "Imports", "LinkingTo"),
    Ncpus = parallel::detectCores()
  )
} else {
  print("Installing binary package on Windows.")
  install.packages(
    pkgs,
    repo = "http://cloud.r-project.org",
    dependencies = c("Depends", "Imports", "LinkingTo"),
    Ncpus = parallel::detectCores(),
    type = "binary"
  )
}
