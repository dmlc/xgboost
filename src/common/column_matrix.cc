/*!
 * Copyright 2017-2022 by XGBoost Contributors
 * \brief Utility for fast column-wise access
 */
#include "column_matrix.h"

namespace xgboost {
namespace common {
void ColumnMatrix::InitStorage(GHistIndexMatrix const& gmat, double sparse_threshold) {
  auto const nfeature = gmat.Features();
  const size_t nrow = gmat.Size();
  // identify type of each column
  type_.resize(nfeature);

  uint32_t max_val = std::numeric_limits<uint32_t>::max();
  for (bst_feature_t fid = 0; fid < nfeature; ++fid) {
    CHECK_LE(gmat.cut.Ptrs()[fid + 1] - gmat.cut.Ptrs()[fid], max_val);
  }

  bool all_dense_column = true;

  std::vector<size_t> feature_counts(nfeature, 0);
  gmat.GetFeatureCounts(feature_counts.data());

  // classify features
  for (bst_feature_t fid = 0; fid < nfeature; ++fid) {
    if (static_cast<double>(feature_counts[fid]) < sparse_threshold * nrow) {
      type_[fid] = kSparseColumn;
      all_dense_column = false;
    } else {
      type_[fid] = kDenseColumn;
    }
  }

  // want to compute storage boundary for each feature
  // using variants of prefix sum scan
  feature_offsets_.resize(nfeature + 1);
  size_t accum_index = 0;
  feature_offsets_[0] = accum_index;
  for (bst_feature_t fid = 1; fid < nfeature + 1; ++fid) {
    if (type_[fid - 1] == kDenseColumn) {
      accum_index += static_cast<size_t>(nrow);
    } else {
      accum_index += feature_counts[fid - 1];
    }
    feature_offsets_[fid] = accum_index;
  }

  SetTypeSize(gmat.MaxNumBinPerFeat());
  auto storage_size =
      feature_offsets_.back() * static_cast<std::underlying_type_t<BinTypeSize>>(bins_type_size_);
  index_.resize(storage_size, 0);
  if (!all_dense_column) {
    row_ind_.resize(feature_offsets_[nfeature]);
  }

  // store least bin id for each feature
  index_base_ = const_cast<uint32_t*>(gmat.cut.Ptrs().data());

  any_missing_ = !gmat.IsDense();

  missing_flags_.clear();
}
}  // namespace common
}  // namespace xgboost
