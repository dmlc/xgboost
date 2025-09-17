/**
 * Copyright 2017-2023, XGBoost Contributors
 * \brief Utility for fast column-wise access
 */
#include "column_matrix.h"

#include <algorithm>    // for transform
#include <cstddef>      // for size_t
#include <cstdint>      // for uint64_t, uint8_t
#include <limits>       // for numeric_limits
#include <type_traits>  // for remove_reference_t
#include <vector>       // for vector

#include "../data/gradient_index.h"  // for GHistIndexMatrix
#include "io.h"                      // for AlignedResourceReadStream, AlignedFileWriteStream
#include "xgboost/base.h"            // for bst_feaature_t
#include "xgboost/span.h"            // for Span

namespace xgboost::common {
void ColumnMatrix::InitStorage(GHistIndexMatrix const& gmat, double sparse_threshold) {
  auto const nfeature = gmat.Features();
  const size_t nrow = gmat.Size();
  // identify type of each column
  type_ = common::MakeFixedVecWithMalloc(nfeature, ColumnType{});

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
  feature_offsets_ = common::MakeFixedVecWithMalloc(nfeature + 1, std::size_t{0});
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

  index_ = common::MakeFixedVecWithMalloc(storage_size, std::uint8_t{0});

  if (!all_dense_column) {
    row_ind_ = common::MakeFixedVecWithMalloc(feature_offsets_[nfeature], std::size_t{0});
  }

  // store least bin id for each feature
  index_base_ = const_cast<uint32_t*>(gmat.cut.Ptrs().data());

  any_missing_ = !gmat.IsDense();

  missing_ = MissingIndicator{0, false};
}

// IO procedures for external memory.
bool ColumnMatrix::Read(AlignedResourceReadStream* fi, uint32_t const* index_base) {
  if (!common::ReadVec(fi, &index_)) {
    return false;
  }
  if (!common::ReadVec(fi, &type_)) {
    return false;
  }
  if (!common::ReadVec(fi, &row_ind_)) {
    return false;
  }
  if (!common::ReadVec(fi, &feature_offsets_)) {
    return false;
  }

  if (!common::ReadVec(fi, &missing_.storage)) {
    return false;
  }
  missing_.InitView();

  index_base_ = index_base;
  if (!fi->Read(&bins_type_size_)) {
    return false;
  }
  if (!fi->Read(&any_missing_)) {
    return false;
  }
  return true;
}

std::size_t ColumnMatrix::Write(AlignedFileWriteStream* fo) const {
  std::size_t bytes{0};

  bytes += common::WriteVec(fo, index_);
  bytes += common::WriteVec(fo, type_);
  bytes += common::WriteVec(fo, row_ind_);
  bytes += common::WriteVec(fo, feature_offsets_);
  bytes += common::WriteVec(fo, missing_.storage);

  bytes += fo->Write(bins_type_size_);
  bytes += fo->Write(any_missing_);

  return bytes;
}
}  // namespace xgboost::common
