## Script used to bootstrap R-universe build.

## Execute git commands to initialize git submodules
system("git submodule init")
system("git submodule update")

manifest <- readLines("tools/cmake-source-files")
manifest <- trimws(manifest)
manifest <- manifest[nzchar(manifest) & !startsWith(manifest, "#")]
embedded_root <- "src"

copy_cmake_source <- function(relative_path) {
  source <- file.path("..", relative_path)
  target <- file.path(embedded_root, relative_path)
  print(paste0("copy: ", source, " -> ", target))
  if (!file.exists(source) && !dir.exists(source)) {
    stop("Missing CMake source manifest entry: ", source)
  }
  dir.create(dirname(target), recursive = TRUE, showWarnings = FALSE)
  if (dir.exists(source)) {
    dir.create(target, recursive = TRUE, showWarnings = FALSE)
    files <- list.files(source, all.files = TRUE, full.names = TRUE)
    files <- files[!basename(files) %in% c(".", "..")]
    if (length(files) && !all(file.copy(files, target, recursive = TRUE))) {
      stop("Failed to copy CMake source directory: ", source)
    }
  } else if (!file.copy(source, target)) {
    stop("Failed to copy CMake source file: ", source)
  }
}

invisible(lapply(manifest, copy_cmake_source))

## license
file.copy("../LICENSE", "./LICENSE")

## misc
path <- file.path("remove_warning_suppression_pragma.sh")
file.remove(path)
path <- file.path("CMakeLists.txt")
file.remove(path)

## remove the directory recursively ./tests/helper_scripts
unlink("tests/helper_scripts", recursive = TRUE)
