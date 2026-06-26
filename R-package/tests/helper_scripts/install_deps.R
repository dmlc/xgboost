## Install dependencies of R package for testing. The list might not be
## up-to-date, check DESCRIPTION for the latest list and update this one if
## inconsistent is found.
ci_pkgs <- c(
  "pkgbuild",
  "roxygen2",
  "XML",
  "cplm",
  "e1071"
)

suggests_pkgs <- c(
  "knitr",
  "rmarkdown",
  "ggplot2",
  "DiagrammeR",
  "DiagrammeRsvg",
  "rsvg",
  "htmlwidgets",
  "Ckmeans.1d.dp",
  "vcd",
  "lintr",
  "testthat",
  "igraph",
  "float",
  "titanic",
  "RhpcBLASctl"
)

imports_pkgs <- c(
  "Matrix",
  "data.table",
  "jsonlite"
)

dependency_scopes <- list(
  ci = ci_pkgs,
  suggests = suggests_pkgs,
  imports = imports_pkgs,
  doc_test = c(imports_pkgs, "DirichletReg", "testthat")
)

scopes <- commandArgs(trailingOnly = TRUE)
if (!length(scopes)) {
  scopes <- c("ci", "suggests", "imports")
}
scopes <- gsub("-", "_", scopes, fixed = TRUE)
if ("all" %in% scopes) {
  scopes <- names(dependency_scopes)
}

unknown_scopes <- setdiff(scopes, names(dependency_scopes))
if (length(unknown_scopes)) {
  stop(
    "Unknown dependency scope(s): ",
    paste(unknown_scopes, collapse = ", "),
    ". Valid scopes are: ",
    paste(c(names(dependency_scopes), "all"), collapse = ", ")
  )
}

pkgs <- unique(unlist(dependency_scopes[scopes], use.names = FALSE))

ncpus <- parallel::detectCores()
print(paste0("Using ", ncpus, " cores to install dependencies."))
print(paste0("Installing dependency scopes: ", paste(scopes, collapse = ", ")))

if (.Platform$OS.type == "unix") {
  print("Installing source packages on unix.")
  install.packages(
    pkgs,
    repos = "https://cloud.r-project.org",
    dependencies = c("Depends", "Imports", "LinkingTo"),
    Ncpus = ncpus
  )
} else {
  print("Installing binary packages on Windows.")
  install.packages(
    pkgs,
    repos = "https://cloud.r-project.org",
    dependencies = c("Depends", "Imports", "LinkingTo"),
    Ncpus = ncpus,
    type = "binary"
  )
}
