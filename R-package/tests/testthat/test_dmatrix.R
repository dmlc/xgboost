library(Matrix)
context("testing xgb.DMatrix functionality")

data(agaricus.test, package = "xgboost")
test_data <- agaricus.test$data[1:100, ]
test_label <- agaricus.test$label[1:100]

n_threads <- 2

test_that("xgb.DMatrix: basic construction", {
  # from sparse matrix
  dtest1 <- xgb.DMatrix(test_data, label = test_label, nthread = n_threads)

  # from dense matrix
  dtest2 <- xgb.DMatrix(as.matrix(test_data), label = test_label, nthread = n_threads)
  expect_equal(getinfo(dtest1, "label"), getinfo(dtest2, "label"))
  expect_equal(dim(dtest1), dim(dtest2))

  # from dense integer matrix
  int_data <- as.matrix(test_data)
  storage.mode(int_data) <- "integer"
  dtest3 <- xgb.DMatrix(int_data, label = test_label, nthread = n_threads)
  expect_equal(dim(dtest1), dim(dtest3))

  n_samples <- 100
  X <- cbind(
    x1 = sample(x = 4, size = n_samples, replace = TRUE),
    x2 = sample(x = 4, size = n_samples, replace = TRUE),
    x3 = sample(x = 4, size = n_samples, replace = TRUE)
  )
  X <- matrix(X, nrow = n_samples)
  y <- rbinom(n = n_samples, size = 1, prob = 1 / 2)

  fd <- xgb.DMatrix(X, label = y, missing = 1, nthread = n_threads)

  dgc <- as(X, "dgCMatrix")
  fdgc <- xgb.DMatrix(dgc, label = y, missing = 1.0, nthread = n_threads)

  dgr <- as(X, "dgRMatrix")
  fdgr <- xgb.DMatrix(dgr, label = y, missing = 1, nthread = n_threads)

  params <- list(tree_method = "hist", nthread = n_threads)
  bst_fd <- xgb.train(
    params, nrounds = 8, fd, watchlist = list(train = fd)
  )
  bst_dgr <- xgb.train(
    params, nrounds = 8, fdgr, watchlist = list(train = fdgr)
  )
  bst_dgc <- xgb.train(
    params, nrounds = 8, fdgc, watchlist = list(train = fdgc)
  )

  raw_fd <- xgb.save.raw(bst_fd, raw_format = "ubj")
  raw_dgr <- xgb.save.raw(bst_dgr, raw_format = "ubj")
  raw_dgc <- xgb.save.raw(bst_dgc, raw_format = "ubj")

  expect_equal(raw_fd, raw_dgr)
  expect_equal(raw_fd, raw_dgc)
})

test_that("xgb.DMatrix: NA", {
  n_samples <- 3
  x <- cbind(
    x1 = sample(x = 4, size = n_samples, replace = TRUE),
    x2 = sample(x = 4, size = n_samples, replace = TRUE)
  )
  x[1, "x1"] <- NA

  m <- xgb.DMatrix(x, nthread = n_threads)
  xgb.DMatrix.save(m, "int.dmatrix")

  x <- matrix(as.numeric(x), nrow = n_samples, ncol = 2)
  colnames(x) <- c("x1", "x2")
  m <- xgb.DMatrix(x, nthread = n_threads)

  xgb.DMatrix.save(m, "float.dmatrix")

  iconn <- file("int.dmatrix", "rb")
  fconn <- file("float.dmatrix", "rb")

  expect_equal(file.size("int.dmatrix"), file.size("float.dmatrix"))

  bytes <- file.size("int.dmatrix")
  idmatrix <- readBin(iconn, "raw", n = bytes)
  fdmatrix <- readBin(fconn, "raw", n = bytes)

  expect_equal(length(idmatrix), length(fdmatrix))
  expect_equal(idmatrix, fdmatrix)

  close(iconn)
  close(fconn)

  file.remove("int.dmatrix")
  file.remove("float.dmatrix")
})

test_that("xgb.DMatrix: saving, loading", {
  # save to a local file
  dtest1 <- xgb.DMatrix(test_data, label = test_label, nthread = n_threads)
  tmp_file <- tempfile('xgb.DMatrix_')
  on.exit(unlink(tmp_file))
  expect_true(xgb.DMatrix.save(dtest1, tmp_file))
  # read from a local file
  expect_output(dtest3 <- xgb.DMatrix(tmp_file), "entries loaded from")
  expect_output(dtest3 <- xgb.DMatrix(tmp_file, silent = TRUE), NA)
  unlink(tmp_file)
  expect_equal(getinfo(dtest1, 'label'), getinfo(dtest3, 'label'))

  # from a libsvm text file
  tmp <- c("0 1:1 2:1", "1 3:1", "0 1:1")
  tmp_file <- tempfile(fileext = ".libsvm")
  writeLines(tmp, tmp_file)
  expect_true(file.exists(tmp_file))
  dtest4 <- xgb.DMatrix(
    paste(tmp_file, "?format=libsvm", sep = ""), silent = TRUE, nthread = n_threads
  )
  expect_equal(dim(dtest4), c(3, 4))
  expect_equal(getinfo(dtest4, 'label'), c(0, 1, 0))

  # check that feature info is saved
  data(agaricus.train, package = 'xgboost')
  dtrain <- xgb.DMatrix(
    data = agaricus.train$data, label = agaricus.train$label, nthread = n_threads
  )
  cnames <- colnames(dtrain)
  expect_equal(length(cnames), 126)
  tmp_file <- tempfile('xgb.DMatrix_')
  xgb.DMatrix.save(dtrain, tmp_file)
  dtrain <- xgb.DMatrix(tmp_file)
  expect_equal(colnames(dtrain), cnames)

  ft <- rep(c("c", "q"), each = length(cnames) / 2)
  setinfo(dtrain, "feature_type", ft)
  expect_equal(ft, getinfo(dtrain, "feature_type"))
})

test_that("xgb.DMatrix: getinfo & setinfo", {
  dtest <- xgb.DMatrix(test_data, nthread = n_threads)
  expect_true(setinfo(dtest, 'label', test_label))
  labels <- getinfo(dtest, 'label')
  expect_equal(test_label, getinfo(dtest, 'label'))

  expect_true(setinfo(dtest, 'label_lower_bound', test_label))
  expect_equal(test_label, getinfo(dtest, 'label_lower_bound'))

  expect_true(setinfo(dtest, 'label_upper_bound', test_label))
  expect_equal(test_label, getinfo(dtest, 'label_upper_bound'))

  expect_true(length(getinfo(dtest, 'weight')) == 0)
  expect_true(length(getinfo(dtest, 'base_margin')) == 0)

  expect_true(setinfo(dtest, 'weight', test_label))
  expect_true(setinfo(dtest, 'base_margin', test_label))
  expect_true(setinfo(dtest, 'group', c(50, 50)))
  expect_error(setinfo(dtest, 'group', test_label))

  # providing character values will give an error
  expect_error(setinfo(dtest, 'weight', rep('a', nrow(test_data))))

  # any other label should error
  expect_error(setinfo(dtest, 'asdf', test_label))
})

test_that("xgb.DMatrix: slice, dim", {
  dtest <- xgb.DMatrix(test_data, label = test_label, nthread = n_threads)
  expect_equal(dim(dtest), dim(test_data))
  dsub1 <- slice(dtest, 1:42)
  expect_equal(nrow(dsub1), 42)
  expect_equal(ncol(dsub1), ncol(test_data))

  dsub2 <- dtest[1:42, ]
  expect_equal(dim(dtest), dim(test_data))
  expect_equal(getinfo(dsub1, 'label'), getinfo(dsub2, 'label'))
})

test_that("xgb.DMatrix: slice, trailing empty rows", {
  data(agaricus.train, package = 'xgboost')
  train_data <- agaricus.train$data
  train_label <- agaricus.train$label
  dtrain <- xgb.DMatrix(
    data = train_data, label = train_label, nthread = n_threads
  )
  slice(dtrain, 6513L)
  train_data[6513, ] <- 0
  dtrain <- xgb.DMatrix(
    data = train_data, label = train_label, nthread = n_threads
  )
  slice(dtrain, 6513L)
  expect_equal(nrow(dtrain), 6513)
})

test_that("xgb.DMatrix: colnames", {
  dtest <- xgb.DMatrix(test_data, label = test_label, nthread = n_threads)
  expect_equal(colnames(dtest), colnames(test_data))
  expect_error(colnames(dtest) <- 'asdf')
  new_names <- make.names(seq_len(ncol(test_data)))
  expect_silent(colnames(dtest) <- new_names)
  expect_equal(colnames(dtest), new_names)
  expect_silent(colnames(dtest) <- NULL)
  expect_null(colnames(dtest))
})

test_that("xgb.DMatrix: nrow is correct for a very sparse matrix", {
  set.seed(123)
  nr <- 1000
  x <- Matrix::rsparsematrix(nr, 100, density = 0.0005)
  # we want it very sparse, so that last rows are empty
  expect_lt(max(x@i), nr)
  dtest <- xgb.DMatrix(x, nthread = n_threads)
  expect_equal(dim(dtest), dim(x))
})

test_that("xgb.DMatrix: print", {
    data(agaricus.train, package = 'xgboost')

    # core DMatrix with just data and labels
    dtrain <- xgb.DMatrix(
      data = agaricus.train$data, label = agaricus.train$label,
      nthread = n_threads
    )
    txt <- capture.output({
        print(dtrain)
    })
    expect_equal(txt, "xgb.DMatrix  dim: 6513 x 126  info: label  colnames: yes")

    # verbose=TRUE prints feature names
    txt <- capture.output({
        print(dtrain, verbose = TRUE)
    })
    expect_equal(txt[[1L]], "xgb.DMatrix  dim: 6513 x 126  info: label  colnames:")
    expect_equal(txt[[2L]], sprintf("'%s'", paste(colnames(dtrain), collapse = "','")))

    # DMatrix with weights and base_margin
    dtrain <- xgb.DMatrix(
      data = agaricus.train$data,
      label = agaricus.train$label,
      weight = seq_along(agaricus.train$label),
      base_margin = agaricus.train$label,
      nthread = n_threads
    )
    txt <- capture.output({
        print(dtrain)
    })
    expect_equal(txt, "xgb.DMatrix  dim: 6513 x 126  info: label weight base_margin  colnames: yes")

    # DMatrix with just features
    dtrain <- xgb.DMatrix(
      data = agaricus.train$data,
      nthread = n_threads
    )
    txt <- capture.output({
        print(dtrain)
    })
    expect_equal(txt, "xgb.DMatrix  dim: 6513 x 126  info: NA  colnames: yes")

    # DMatrix with no column names
    data_no_colnames <- agaricus.train$data
    colnames(data_no_colnames) <- NULL
    dtrain <- xgb.DMatrix(
      data = data_no_colnames,
      nthread = n_threads
    )
    txt <- capture.output({
        print(dtrain)
    })
    expect_equal(txt, "xgb.DMatrix  dim: 6513 x 126  info: NA  colnames: no")
})

test_that("xgb.DMatrix: Inf as missing", {
  x_inf <- matrix(as.numeric(1:10), nrow = 5)
  x_inf[2, 1] <- Inf

  x_nan <- x_inf
  x_nan[2, 1] <- NA_real_

  m_inf <- xgb.DMatrix(x_inf, nthread = n_threads, missing = Inf)
  xgb.DMatrix.save(m_inf, "inf.dmatrix")

  m_nan <- xgb.DMatrix(x_nan, nthread = n_threads, missing = NA_real_)
  xgb.DMatrix.save(m_nan, "nan.dmatrix")

  infconn <- file("inf.dmatrix", "rb")
  nanconn <- file("nan.dmatrix", "rb")

  expect_equal(file.size("inf.dmatrix"), file.size("nan.dmatrix"))

  bytes <- file.size("inf.dmatrix")
  infdmatrix <- readBin(infconn, "raw", n = bytes)
  nandmatrix <- readBin(nanconn, "raw", n = bytes)

  expect_equal(length(infdmatrix), length(nandmatrix))
  expect_equal(infdmatrix, nandmatrix)

  close(infconn)
  close(nanconn)

  file.remove("inf.dmatrix")
  file.remove("nan.dmatrix")
})

test_that("xgb.DMatrix: error on three-dimensional array", {
  set.seed(123)
  x <- matrix(rnorm(500), nrow = 50)
  y <- rnorm(400)
  dim(y) <- c(50, 4, 2)
  expect_error(xgb.DMatrix(data = x, label = y))
})

test_that("xgb.DMatrix: can get group for both 'qid' and 'group' constructors", {
  set.seed(123)
  x <- matrix(rnorm(1000), nrow = 100)
  group <- c(20, 20, 60)
  qid <- c(rep(1, 20), rep(2, 20), rep(3, 60))

  gr_mat <- xgb.DMatrix(x, group = group)
  qid_mat <- xgb.DMatrix(x, qid = qid)

  info_gr <- getinfo(gr_mat, "group")
  info_qid <- getinfo(qid_mat, "group")
  expect_equal(info_gr, info_qid)

  expected_gr <- c(0, 20, 40, 100)
  expect_equal(info_gr, expected_gr)
})

test_that("xgb.DMatrix: data.frame", {
  df <- data.frame(
    a = (1:4) / 10,
    num = c(1, NA, 3, 4),
    as.int = as.integer(c(1, 2, 3, 4)),
    lo = c(TRUE, FALSE, NA, TRUE),
    str.fac = c("a", "b", "d", "c"),
    as.fac = as.factor(c(3, 5, 8, 11)),
    stringsAsFactors = TRUE
  )

  m <- xgb.DMatrix(df, enable_categorical = TRUE)
  expect_equal(colnames(m), colnames(df))
  expect_equal(
    getinfo(m, "feature_type"), c("float", "float", "int", "i", "c", "c")
  )
  expect_error(xgb.DMatrix(df))

  df <- data.frame(
    missing = c("a", "b", "d", NA),
    valid = c("a", "b", "d", "c"),
    stringsAsFactors = TRUE
  )
  m <- xgb.DMatrix(df, enable_categorical = TRUE)
  expect_equal(getinfo(m, "feature_type"), c("c", "c"))
})

test_that("xgb.DMatrix: can take multi-dimensional 'base_margin'", {
  set.seed(123)
  x <- matrix(rnorm(100 * 10), nrow = 100)
  y <- matrix(rnorm(100 * 2), nrow = 100)
  b <- matrix(rnorm(100 * 2), nrow = 100)
  model <- xgb.train(
    data = xgb.DMatrix(data = x, label = y, nthread = n_threads),
    params = list(
      objective = "reg:squarederror",
      tree_method = "hist",
      multi_strategy = "multi_output_tree",
      base_score = 0,
      nthread = n_threads
    ),
    nround = 1
  )
  pred_only_x <- predict(model, x, nthread = n_threads, reshape = TRUE)
  pred_w_base <- predict(
    model,
    xgb.DMatrix(data = x, base_margin = b, nthread = n_threads),
    nthread = n_threads,
    reshape = TRUE
  )
  expect_equal(pred_only_x, pred_w_base - b, tolerance = 1e-5)
})

test_that("xgb.DMatrix: number of non-missing matches data", {
  x <- matrix(1:10, nrow = 5)
  dm1 <- xgb.DMatrix(x)
  expect_equal(xgb.get.DMatrix.num.non.missing(dm1), 10)

  x[2, 2] <- NA
  x[4, 1] <- NA
  dm2 <- xgb.DMatrix(x)
  expect_equal(xgb.get.DMatrix.num.non.missing(dm2), 8)
})

test_that("xgb.DMatrix: retrieving data as CSR", {
  data(mtcars)
  dm <- xgb.DMatrix(as.matrix(mtcars))
  csr <- xgb.get.DMatrix.data(dm)
  expect_equal(dim(csr), dim(mtcars))
  expect_equal(colnames(csr), colnames(mtcars))
  expect_equal(unname(as.matrix(csr)), unname(as.matrix(mtcars)), tolerance = 1e-6)
})

test_that("xgb.DMatrix: quantile cuts look correct", {
  data(mtcars)
  y <- mtcars$mpg
  x <- as.matrix(mtcars[, -1])
  dm <- xgb.DMatrix(x, label = y)
  model <- xgb.train(
    data = dm,
    params = list(
      tree_method = "hist",
      max_bin = 8,
      nthread = 1
    ),
    nrounds = 3
  )
  qcut_list <- xgb.get.DMatrix.qcut(dm, "list")
  qcut_arrays <- xgb.get.DMatrix.qcut(dm, "arrays")

  expect_equal(length(qcut_arrays), 2)
  expect_equal(names(qcut_arrays), c("indptr", "data"))
  expect_equal(length(qcut_arrays$indptr), ncol(x) + 1)
  expect_true(min(diff(qcut_arrays$indptr)) > 0)

  col_min <- apply(x, 2, min)
  col_max <- apply(x, 2, max)

  expect_equal(length(qcut_list), ncol(x))
  expect_equal(names(qcut_list), colnames(x))
  lapply(
    seq(1, ncol(x)),
    function(col) {
      cuts <- qcut_list[[col]]
      expect_true(min(diff(cuts)) > 0)
      expect_true(col_min[col] > cuts[1])
      expect_true(col_max[col] < cuts[length(cuts)])
      expect_true(length(cuts) <= 9)
    }
  )
})
