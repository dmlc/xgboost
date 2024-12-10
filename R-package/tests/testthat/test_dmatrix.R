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
    params, nrounds = 8, fd, evals = list(train = fd), verbose = 0
  )
  bst_dgr <- xgb.train(
    params, nrounds = 8, fdgr, evals = list(train = fdgr), verbose = 0
  )
  bst_dgc <- xgb.train(
    params, nrounds = 8, fdgc, evals = list(train = fdgc), verbose = 0
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
  fname_int <- file.path(tempdir(), "int.dmatrix")
  xgb.DMatrix.save(m, fname_int)

  x <- matrix(as.numeric(x), nrow = n_samples, ncol = 2)
  colnames(x) <- c("x1", "x2")
  m <- xgb.DMatrix(x, nthread = n_threads)

  fname_float <- file.path(tempdir(), "float.dmatrix")
  xgb.DMatrix.save(m, fname_float)

  iconn <- file(fname_int, "rb")
  fconn <- file(fname_float, "rb")

  expect_equal(file.size(fname_int), file.size(fname_float))

  bytes <- file.size(fname_int)
  idmatrix <- readBin(iconn, "raw", n = bytes)
  fdmatrix <- readBin(fconn, "raw", n = bytes)

  expect_equal(length(idmatrix), length(fdmatrix))
  expect_equal(idmatrix, fdmatrix)

  close(iconn)
  close(fconn)

  file.remove(fname_int)
  file.remove(fname_float)
})

test_that("xgb.DMatrix: saving, loading", {
  # save to a local file
  dtest1 <- xgb.DMatrix(test_data, label = test_label, nthread = n_threads)
  tmp_file <- tempfile('xgb.DMatrix_')
  on.exit(unlink(tmp_file))
  expect_true(xgb.DMatrix.save(dtest1, tmp_file))
  # read from a local file
  xgb.set.config(verbosity = 2)
  expect_output(dtest3 <- xgb.DMatrix(tmp_file), "entries loaded from")
  xgb.set.config(verbosity = 1)
  expect_output(dtest3 <- xgb.DMatrix(tmp_file), NA)
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
  xgb.set.config(verbosity = 0)
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
  dsub1 <- xgb.slice.DMatrix(dtest, 1:42)
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
  xgb.slice.DMatrix(dtrain, 6513L)
  train_data[6513, ] <- 0
  dtrain <- xgb.DMatrix(
    data = train_data, label = train_label, nthread = n_threads
  )
  xgb.slice.DMatrix(dtrain, 6513L)
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
    expect_equal(txt, "xgb.DMatrix  dim: 6513 x 126  info: base_margin, label, weight  colnames: yes")

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
  fname_inf <- file.path(tempdir(), "inf.dmatrix")
  xgb.DMatrix.save(m_inf, fname_inf)

  m_nan <- xgb.DMatrix(x_nan, nthread = n_threads, missing = NA_real_)
  fname_nan <- file.path(tempdir(), "nan.dmatrix")
  xgb.DMatrix.save(m_nan, fname_nan)

  infconn <- file(fname_inf, "rb")
  nanconn <- file(fname_nan, "rb")

  expect_equal(file.size(fname_inf), file.size(fname_nan))

  bytes <- file.size(fname_inf)
  infdmatrix <- readBin(infconn, "raw", n = bytes)
  nandmatrix <- readBin(nanconn, "raw", n = bytes)

  expect_equal(length(infdmatrix), length(nandmatrix))
  expect_equal(infdmatrix, nandmatrix)

  close(infconn)
  close(nanconn)

  file.remove(fname_inf)
  file.remove(fname_nan)
})

test_that("xgb.DMatrix: missing in CSR", {
  x_dense <- matrix(as.numeric(1:10), nrow = 5)
  x_dense[2, 1] <- NA_real_

  x_csr <- as(x_dense, "RsparseMatrix")

  m_dense <- xgb.DMatrix(x_dense, nthread = n_threads, missing = NA_real_)
  xgb.DMatrix.save(m_dense, "dense.dmatrix")

  m_csr <- xgb.DMatrix(x_csr, nthread = n_threads, missing = NA)
  xgb.DMatrix.save(m_csr, "csr.dmatrix")

  denseconn <- file("dense.dmatrix", "rb")
  csrconn <- file("csr.dmatrix", "rb")

  expect_equal(file.size("dense.dmatrix"), file.size("csr.dmatrix"))

  bytes <- file.size("dense.dmatrix")
  densedmatrix <- readBin(denseconn, "raw", n = bytes)
  csrmatrix <- readBin(csrconn, "raw", n = bytes)

  expect_equal(length(densedmatrix), length(csrmatrix))
  expect_equal(densedmatrix, csrmatrix)

  close(denseconn)
  close(csrconn)

  file.remove("dense.dmatrix")
  file.remove("csr.dmatrix")
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

  m <- xgb.DMatrix(df)
  expect_equal(colnames(m), colnames(df))
  expect_equal(
    getinfo(m, "feature_type"), c("float", "float", "int", "i", "c", "c")
  )

  df <- data.frame(
    missing = c("a", "b", "d", NA),
    valid = c("a", "b", "d", "c"),
    stringsAsFactors = TRUE
  )
  m <- xgb.DMatrix(df)
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
  pred_only_x <- predict(model, x)
  pred_w_base <- predict(
    model,
    xgb.DMatrix(data = x, base_margin = b)
  )
  expect_equal(pred_only_x, pred_w_base - b, tolerance = 1e-5)
})

test_that("xgb.DMatrix: QuantileDMatrix produces same result as DMatrix", {
  data(mtcars)
  y <- mtcars[, 1]
  x <- mtcars[, -1]

  cast_matrix <- function(x) as.matrix(x)
  cast_df <- function(x) as.data.frame(x)
  cast_csr <- function(x) as(as.matrix(x), "RsparseMatrix")
  casting_funs <- list(cast_matrix, cast_df, cast_csr)

  for (casting_fun in casting_funs) {

    qdm <- xgb.QuantileDMatrix(
      data = casting_fun(x),
      label = y,
      nthread = n_threads,
      max_bin = 5
    )
    params <- list(
      tree_method = "hist",
      objective = "reg:squarederror",
      nthread = n_threads,
      max_bin = 5
    )
    model_qdm <- xgb.train(
      params = params,
      data = qdm,
      nrounds = 2
    )
    pred_qdm <- predict(model_qdm, x)

    dm <- xgb.DMatrix(
      data = x,
      label = y,
      nthread = n_threads
    )
    model_dm <- xgb.train(
      params = params,
      data = dm,
      nrounds = 2
    )
    pred_dm <- predict(model_dm, x)

    expect_equal(pred_qdm, pred_dm)
  }
})

test_that("xgb.DMatrix: QuantileDMatrix is not accepted by exact method", {
  data(mtcars)
  y <- mtcars[, 1]
  x <- as.matrix(mtcars[, -1])
  qdm <- xgb.QuantileDMatrix(
    data = x,
    label = y,
    nthread = n_threads
  )
  params <- list(
    tree_method = "exact",
    objective = "reg:squarederror",
    nthread = n_threads
  )
  expect_error({
    xgb.train(
      params = params,
      data = qdm,
      nrounds = 2
    )
  })
})

test_that("xgb.DMatrix: ExtMemDMatrix produces the same results as regular DMatrix", {
  data(mtcars)
  y <- mtcars[, 1]
  x <- as.matrix(mtcars[, -1])
  set.seed(123)
  params <- list(
    objective = "reg:squarederror",
    nthread = n_threads
  )
  model <- xgb.train(
    data = xgb.DMatrix(x, label = y),
    params = params,
    nrounds = 5
  )
  pred <- predict(model, x)
  pred <- unname(pred)

  iterator_env <- as.environment(
    list(
      iter = 0,
      x = mtcars[, -1],
      y = mtcars[, 1]
    )
  )
  iterator_next <- function(iterator_env) {
    curr_iter <- iterator_env[["iter"]]
    if (curr_iter >= 2) {
      return(NULL)
    }
    if (curr_iter == 0) {
      x_batch <- iterator_env[["x"]][1:16, ]
      y_batch <- iterator_env[["y"]][1:16]
    } else {
      x_batch <- iterator_env[["x"]][17:32, ]
      y_batch <- iterator_env[["y"]][17:32]
    }
    on.exit({
      iterator_env[["iter"]] <- curr_iter + 1
    })
    return(xgb.DataBatch(data = x_batch, label = y_batch))
  }
  iterator_reset <- function(iterator_env) {
    iterator_env[["iter"]] <- 0
  }
  data_iterator <- xgb.DataIter(
    env = iterator_env,
    f_next = iterator_next,
    f_reset = iterator_reset
  )
  cache_prefix <- tempdir()
  edm <- xgb.ExtMemDMatrix(data_iterator, cache_prefix, nthread = 1)
  expect_true(inherits(edm, "xgb.ExtMemDMatrix"))
  expect_true(inherits(edm, "xgb.DMatrix"))
  set.seed(123)
  model_ext <- xgb.train(
    data = edm,
    params = params,
    nrounds = 5
  )

  pred_model1_edm <- predict(model, edm)
  pred_model2_mat <- predict(model_ext, x) |> unname()
  pred_model2_edm <- predict(model_ext, edm)

  expect_equal(pred_model1_edm, pred)
  expect_equal(pred_model2_mat, pred)
  expect_equal(pred_model2_edm, pred)
})

test_that("xgb.DMatrix: External QDM produces same results as regular QDM", {
  data(mtcars)
  y <- mtcars[, 1]
  x <- as.matrix(mtcars[, -1])
  set.seed(123)
  params <- list(
    objective = "reg:squarederror",
    nthread = n_threads,
    max_bin = 3
  )
  model <- xgb.train(
    data = xgb.QuantileDMatrix(
      x,
      label = y,
      nthread = 1,
      max_bin = 3
    ),
    params = params,
    nrounds = 5
  )
  pred <- predict(model, x)
  pred <- unname(pred)

  iterator_env <- as.environment(
    list(
      iter = 0,
      x = mtcars[, -1],
      y = mtcars[, 1]
    )
  )
  iterator_next <- function(iterator_env) {
    curr_iter <- iterator_env[["iter"]]
    if (curr_iter >= 2) {
      return(NULL)
    }
    if (curr_iter == 0) {
      x_batch <- iterator_env[["x"]][1:16, ]
      y_batch <- iterator_env[["y"]][1:16]
    } else {
      x_batch <- iterator_env[["x"]][17:32, ]
      y_batch <- iterator_env[["y"]][17:32]
    }
    on.exit({
      iterator_env[["iter"]] <- curr_iter + 1
    })
    return(xgb.DataBatch(data = x_batch, label = y_batch))
  }
  iterator_reset <- function(iterator_env) {
    iterator_env[["iter"]] <- 0
  }
  data_iterator <- xgb.DataIter(
    env = iterator_env,
    f_next = iterator_next,
    f_reset = iterator_reset
  )
  cache_prefix <- tempdir()
  qdm <- xgb.QuantileDMatrix.from_iterator(
    data_iterator,
    max_bin = 3,
    nthread = 1
  )
  expect_true(inherits(qdm, "xgb.QuantileDMatrix"))
  expect_true(inherits(qdm, "xgb.DMatrix"))
  set.seed(123)
  model_ext <- xgb.train(
    data = qdm,
    params = params,
    nrounds = 5
  )

  pred_model1_qdm <- predict(model, qdm)
  pred_model2_mat <- predict(model_ext, x) |> unname()
  pred_model2_qdm <- predict(model_ext, qdm)

  expect_equal(pred_model1_qdm, pred)
  expect_equal(pred_model2_mat, pred)
  expect_equal(pred_model2_qdm, pred)
})

test_that("xgb.DMatrix: R errors thrown on DataIterator are thrown back to the user", {
  data(mtcars)
  iterator_env <- as.environment(
    list(
      iter = 0,
      x = mtcars[, -1],
      y = mtcars[, 1]
    )
  )
  iterator_next <- function(iterator_env) {
    curr_iter <- iterator_env[["iter"]]
    if (curr_iter >= 2) {
      return(0)
    }
    if (curr_iter == 0) {
      x_batch <- iterator_env[["x"]][1:16, ]
      y_batch <- iterator_env[["y"]][1:16]
    } else {
      stop("custom error")
    }
    on.exit({
      iterator_env[["iter"]] <- curr_iter + 1
    })
    return(xgb.DataBatch(data = x_batch, label = y_batch))
  }
  iterator_reset <- function(iterator_env) {
    iterator_env[["iter"]] <- 0
  }
  data_iterator <- xgb.DataIter(
    env = iterator_env,
    f_next = iterator_next,
    f_reset = iterator_reset
  )
  expect_error(
    {xgb.ExtMemDMatrix(data_iterator, nthread = 1)},
    "custom error"
  )
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

test_that("xgb.DMatrix: slicing keeps field indicators", {
  data(mtcars)
  x <- as.matrix(mtcars[, -1])
  y <- mtcars[, 1]
  dm <- xgb.DMatrix(
    data = x,
    label_lower_bound = -y,
    label_upper_bound = y,
    nthread = 1
  )
  idx_take <- seq(1, 5)
  dm_slice <- xgb.slice.DMatrix(dm, idx_take)

  expect_true(xgb.DMatrix.hasinfo(dm_slice, "label_lower_bound"))
  expect_true(xgb.DMatrix.hasinfo(dm_slice, "label_upper_bound"))
  expect_false(xgb.DMatrix.hasinfo(dm_slice, "label"))

  expect_equal(getinfo(dm_slice, "label_lower_bound"), -y[idx_take], tolerance = 1e-6)
  expect_equal(getinfo(dm_slice, "label_upper_bound"), y[idx_take], tolerance = 1e-6)
})

test_that("xgb.DMatrix: can slice with groups", {
  data(iris)
  x <- as.matrix(iris[, -5])
  set.seed(123)
  y <- sample(3, size = nrow(x), replace = TRUE)
  group <- c(50, 50, 50)
  dm <- xgb.DMatrix(x, label = y, group = group, nthread = 1)
  idx_take <- seq(1, 50)
  dm_slice <- xgb.slice.DMatrix(dm, idx_take, allow_groups = TRUE)

  expect_true(xgb.DMatrix.hasinfo(dm_slice, "label"))
  expect_false(xgb.DMatrix.hasinfo(dm_slice, "group"))
  expect_false(xgb.DMatrix.hasinfo(dm_slice, "qid"))
  expect_null(getinfo(dm_slice, "group"))
  expect_equal(getinfo(dm_slice, "label"), y[idx_take], tolerance = 1e-6)
})

test_that("xgb.DMatrix: can read CSV", {
  txt <- paste(
    "1,2,3",
    "-1,3,2",
    sep = "\n"
  )
  fname <- file.path(tempdir(), "data.csv")
  writeChar(txt, fname)
  uri <- paste0(fname, "?format=csv&label_column=0")
  dm <- xgb.DMatrix(uri, silent = TRUE)
  expect_equal(getinfo(dm, "label"), c(1, -1))
  expect_equal(
    as.matrix(xgb.get.DMatrix.data(dm)),
    matrix(c(2, 3, 3, 2), nrow = 2, byrow = TRUE)
  )
})
