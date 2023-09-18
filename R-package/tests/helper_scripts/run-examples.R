## Helper script for running individual examples.
library(pkgload)
library(xgboost)

files <- list.files("./man")


run_example_timeit <- function(f) {
  path <- paste("./man/", f, sep = "")
  print(paste("Test", f))
  flush.console()
  t0 <- proc.time()
  run_example(path)
  t1 <- proc.time()
  list(file = f, time = t1 - t0)
}

timings <- lapply(files, run_example_timeit)

for (t in timings) {
  ratio <- t$time[1] / t$time[3]
  if (!is.na(ratio) && !is.infinite(ratio) && ratio >= 2.5) {
    print(paste("Offending example:", t$file, ratio))
  }
}
