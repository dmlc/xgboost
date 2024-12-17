#' Parse model text dump
#'
#' Parse a boosted tree model text dump into a `data.table` structure.
#'
#' Note that this function does not work with models that were fitted to
#' categorical data, and is only applicable to tree-based boosters (not `gblinear`).
#' @param model Object of class `xgb.Booster`. If it contains feature names (they can
#'   be set through [setinfo()]), they will be used in the output from this function.
#'
#'   If the model contains categorical features, an error will be thrown.
#' @param trees An integer vector of (base-1) tree indices that should be used. The default
#'   (`NULL`) uses all trees. Useful, e.g., in multiclass classification to get only
#'   the trees of one class.
#' @param use_int_id A logical flag indicating whether nodes in columns "Yes", "No", and
#'   "Missing" should be represented as integers (when `TRUE`) or as "Tree-Node"
#'   character strings (when `FALSE`, default).
#' @inheritParams xgb.train
#' @return
#' A `data.table` with detailed information about tree nodes. It has the following columns:
#' - `Tree`: integer ID of a tree in a model (zero-based index).
#' - `Node`: integer ID of a node in a tree (zero-based index).
#' - `ID`: character identifier of a node in a model (only when `use_int_id = FALSE`).
#' - `Feature`: for a branch node, a feature ID or name (when available);
#'              for a leaf node, it simply labels it as `"Leaf"`.
#' - `Split`: location of the split for a branch node (split condition is always "less than").
#' - `Yes`: ID of the next node when the split condition is met.
#' - `No`: ID of the next node when the split condition is not met.
#' - `Missing`: ID of the next node when the branch value is missing.
#' - `Gain`: either the split gain (change in loss) or the leaf value.
#' - `Cover`: metric related to the number of observations either seen by a split
#'            or collected by a leaf during training.
#'
#' When `use_int_id = FALSE`, columns "Yes", "No", and "Missing" point to model-wide node identifiers
#' in the "ID" column. When `use_int_id = TRUE`, those columns point to node identifiers from
#' the corresponding trees in the "Node" column.
#'
#' @examples
#' # Basic use:
#'
#' data(agaricus.train, package = "xgboost")
#' ## Keep the number of threads to 1 for examples
#' nthread <- 1
#' data.table::setDTthreads(nthread)
#'
#' bst <- xgb.train(
#'   data = xgb.DMatrix(agaricus.train$data, label = agaricus.train$label),
#'   nrounds = 2,
#'   params = xgb.params(
#'     max_depth = 2,
#'     nthread = nthread,
#'     objective = "binary:logistic"
#'   )
#' )
#'
#' # This bst model already has feature_names stored with it, so those would be used when
#' # feature_names is not set:
#' dt <- xgb.model.dt.tree(bst)
#'
#' # How to match feature names of splits that are following a current 'Yes' branch:
#' merge(
#'   dt,
#'   dt[, .(ID, Y.Feature = Feature)], by.x = "Yes", by.y = "ID", all.x = TRUE
#' )[
#'   order(Tree, Node)
#' ]
#'
#' @export
xgb.model.dt.tree <- function(model, trees = NULL, use_int_id = FALSE, ...) {
  check.deprecation(deprecated_dttree_params, match.call(), ...)

  if (!inherits(model, "xgb.Booster")) {
    stop("Either 'model' must be an object of class xgb.Booster")
  }

  if (xgb.has_categ_features(model)) {
    stop("Cannot produce tables for models having categorical features.")
  }

  if (!is.null(trees)) {
    if (!is.vector(trees) || (!is.numeric(trees) && !is.integer(trees))) {
      stop("trees: must be a vector of integers.")
    }
    trees <- trees - 1L
    if (anyNA(trees) || min(trees) < 0) {
      stop("Passed invalid tree indices.")
    }
  }

  feature_names <- NULL
  if (inherits(model, "xgb.Booster")) {
    feature_names <- xgb.feature_names(model)
  }

  text <- xgb.dump(model = model, with_stats = TRUE)

  if (length(text) < 2 || !any(grepl('leaf=(-?\\d+)', text))) {
    stop("Non-tree model detected! This function can only be used with tree models.")
  }

  position <- which(grepl("booster", text, fixed = TRUE))

  add.tree.id <- function(node, tree) if (use_int_id) node else paste(tree, node, sep = "-")

  anynumber_regex <- "[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?"

  td <- data.table(t = text)
  td[position, Tree := 1L]
  td[, Tree := cumsum(ifelse(is.na(Tree), 0L, Tree)) - 1L]

  if (is.null(trees)) {
    trees <- 0:max(td$Tree)
  } else {
    trees <- trees[trees >= 0 & trees <= max(td$Tree)]
  }
  td <- td[Tree %in% trees & !is.na(t) & !startsWith(t, 'booster')]

  td[, Node := as.integer(sub("^([0-9]+):.*", "\\1", t))]
  if (!use_int_id) td[, ID := add.tree.id(Node, Tree)]
  td[, isLeaf := grepl("leaf", t, fixed = TRUE)]

  # parse branch lines
  branch_rx_nonames <- paste0("f(\\d+)<(", anynumber_regex, ")\\] yes=(\\d+),no=(\\d+),missing=(\\d+),",
                              "gain=(", anynumber_regex, "),cover=(", anynumber_regex, ")")
  branch_rx_w_names <- paste0("\\d+:\\[(.+)<(", anynumber_regex, ")\\] yes=(\\d+),no=(\\d+),missing=(\\d+),",
                              "gain=(", anynumber_regex, "),cover=(", anynumber_regex, ")")
  text_has_feature_names <- FALSE
  if (NROW(feature_names)) {
    branch_rx <- branch_rx_w_names
    text_has_feature_names <- TRUE
  } else {
    branch_rx <- branch_rx_nonames
  }
  branch_cols <- c("Feature", "Split", "Yes", "No", "Missing", "Gain", "Cover")
  td[
    isLeaf == FALSE,
    (branch_cols) := {
      matches <- regmatches(t, regexec(branch_rx, t))
      # skip some indices with spurious capture groups from anynumber_regex
      xtr <- do.call(rbind, matches)[, c(2, 3, 5, 6, 7, 8, 10), drop = FALSE]
      xtr[, 3:5] <- add.tree.id(xtr[, 3:5], Tree)
      if (length(xtr) == 0) {
        as.data.table(
          list(Feature = "NA", Split = "NA", Yes = "NA", No = "NA", Missing = "NA", Gain = "NA", Cover = "NA")
        )
      } else {
        as.data.table(xtr)
      }
    }
  ]

  # assign feature_names when available
  is_stump <- function() {
    return(length(td$Feature) == 1 && is.na(td$Feature))
  }
  if (!text_has_feature_names) {
    if (!is.null(feature_names) && !is_stump()) {
      if (length(feature_names) <= max(as.numeric(td$Feature), na.rm = TRUE))
        stop("feature_names has less elements than there are features used in the model")
      td[isLeaf == FALSE, Feature := feature_names[as.numeric(Feature) + 1]]
    }
  }

  # parse leaf lines
  leaf_rx <- paste0("leaf=(", anynumber_regex, "),cover=(", anynumber_regex, ")")
  leaf_cols <- c("Feature", "Gain", "Cover")
  td[
    isLeaf == TRUE,
    (leaf_cols) := {
      matches <- regmatches(t, regexec(leaf_rx, t))
      xtr <- do.call(rbind, matches)[, c(2, 4)]
      if (length(xtr) == 2) {
        c("Leaf", as.data.table(xtr[1]), as.data.table(xtr[2]))
      } else {
        c("Leaf", as.data.table(xtr))
      }
    }
  ]

  # convert some columns to numeric
  numeric_cols <- c("Split", "Gain", "Cover")
  td[, (numeric_cols) := lapply(.SD, as.numeric), .SDcols = numeric_cols]
  if (use_int_id) {
    int_cols <- c("Yes", "No", "Missing")
    td[, (int_cols) := lapply(.SD, as.integer), .SDcols = int_cols]
  }

  td[, t := NULL]
  td[, isLeaf := NULL]

  td[order(Tree, Node)]
}

# Avoid notes during CRAN check.
# The reason is that these variables are never declared
# They are mainly column names inferred by Data.table...
globalVariables(c("Tree", "Node", "ID", "Feature", "t", "isLeaf", ".SD", ".SDcols"))
