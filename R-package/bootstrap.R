## Script used to bootstrap R-universe build.

## Execute git commands to initialize git submodules
system("git submodule init")
system("git submodule update")

## core
file.copy("../src", "./src/", recursive = TRUE)
file.copy("../include", "./src/", recursive = TRUE)
file.copy("../amalgamation", "./src/", recursive = TRUE)

## dmlc-core
dir.create("./src/dmlc-core")
file.copy("../dmlc-core/include", "./src/dmlc-core/", recursive = TRUE)
file.copy("../dmlc-core/src", "./src/dmlc-core/", recursive = TRUE)

pkgroot <- function(path) {
  ## read the file from path, replace the PKGROOT=../../ with PKGROOT=.
  lines <- readLines(path)
  lines <- gsub("PKGROOT=../../", "PKGROOT=.", lines, fixed = TRUE)
  writeLines(lines, path)
}

## makefile and license
file.copy("../LICENSE", "./LICENSE")
pkgroot("./src/Makevars.in")
pkgroot("./src/Makevars.win.in")

## misc
path <- file.path("remove_warning_suppression_pragma.sh")
file.remove(path)
path <- file.path("CMakeLists.txt")
file.remove(path)

## remove the directory recursively ./tests/helper_scripts
unlink("tests/helper_scripts", recursive = TRUE)
