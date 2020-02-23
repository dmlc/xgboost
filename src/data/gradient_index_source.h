/*!
 * Copyright 2017-2020 by XGBoost Contributors
 */
#ifndef XGBOOST_DATA_GRADIENT_INDEX_SOURCE_H_
#define XGBOOST_DATA_GRADIENT_INDEX_SOURCE_H_

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <vector>

#include "xgboost/base.h"
#include "xgboost/data.h"
#include "../common/common.h"
#include "../common/hist_util.h"

namespace xgboost {

namespace common {


/*! \brief column type */
enum ColumnType {
  kDenseColumn,
  kSparseColumn
};

/*! \brief a column storage, to be used with ApplySplit. Note that each
    bin id is stored as index[i] + index_base. */
class Column {
 public:
  Column(ColumnType type, const uint32_t* index, uint32_t index_base,
         const size_t* row_ind, size_t len)
      : type_(type),
        index_(index),
        index_base_(index_base),
        row_ind_(row_ind),
        len_(len) {}
  size_t Size() const { return len_; }
  uint32_t GetGlobalBinIdx(size_t idx) const {
    if (IsMissing(idx)) {
      return std::numeric_limits<uint32_t>::max();
    }
    return index_base_ + index_[idx];
  }
  uint32_t GetFeatureBinIdx(size_t idx) const { return index_[idx]; }
  // column.GetFeatureBinIdx(idx) + column.GetBaseIdx(idx) ==
  // column.GetGlobalBinIdx(idx)
  uint32_t GetBaseIdx() const { return index_base_; }
  ColumnType GetType() const { return type_; }
  size_t GetRowIdx(size_t idx) const {
    // clang-tidy worries that row_ind_ might be a nullptr, which is possible,
    // but low level structure is not safe anyway.
    return type_ == ColumnType::kDenseColumn ? idx : row_ind_[idx];  // NOLINT
  }
  bool IsMissing(size_t idx) const {
    return index_[idx] == std::numeric_limits<uint32_t>::max();
  }
  const size_t* GetRowData() const { return row_ind_; }
  const uint32_t* Index() const {
    return index_;
  }

 private:
  ColumnType type_;
  const uint32_t* index_;
  uint32_t index_base_;
  const size_t* row_ind_;
  const size_t len_;
};

/*! \brief a collection of columns, with support for construction from
    GHistIndexMatrix. */
class ColumnMatrix {
 public:
  // get number of features
  inline bst_uint GetNumFeature() const {
    return static_cast<bst_uint>(type_.size());
  }

  // construct column matrix from GHistIndexMatrix
  void Init(const GradientIndexPage& gmat, double  sparse_threshold);

  /* Fetch an individual column. This code should be used with XGBOOST_TYPE_SWITCH
     to determine type of bin id's */
  inline Column GetColumn(unsigned fid) const {
    Column c(type_[fid], &index_[boundary_[fid].index_begin], index_base_[fid],
             (type_[fid] == ColumnType::kSparseColumn ?
              &row_ind_[boundary_[fid].row_ind_begin] : nullptr),
             boundary_[fid].index_end - boundary_[fid].index_begin);
    return c;
  }

 private:
  struct ColumnBoundary {
    // indicate where each column's index and row_ind is stored.
    // index_begin and index_end are logical offsets, so they should be converted to
    // actual offsets by scaling with packing_factor_
    size_t index_begin;
    size_t index_end;
    size_t row_ind_begin;
    size_t row_ind_end;
  };

  std::vector<size_t> feature_counts_;
  std::vector<ColumnType> type_;
  std::vector<uint32_t> index_;  // index_: may store smaller integers; needs padding
  std::vector<size_t> row_ind_;
  std::vector<ColumnBoundary> boundary_;

  // index_base_[fid]: least bin id for feature fid
  std::vector<uint32_t> index_base_;
};
}  // namespace common

/*!
 * \brief preprocessed gradient index matrix, in CSR format Transform floating values to
 *  integer index in histogram This is a gradient histogram index.
 */
struct GradientIndexPage {
  /*! \brief row pointer to rows by element position */
  std::vector<size_t> row_ptr;
  /*! \brief The index data */
  std::vector<uint32_t> index;
  /*! \brief hit count of each index */
  std::vector<size_t> hit_count;
  /*! \brief The corresponding cuts */
  common::HistogramCuts cut;
  common::ColumnMatrix column_matrix;

  GradientIndexPage() = default;

  GradientIndexPage(DMatrix *p_fmat, BatchParam const& p) {
    this->Init(p_fmat, p.max_bin);
  }

  // Create a global histogram matrix, given cut
  void Init(DMatrix *p_fmat, int max_num_bins);
  GradientIndexPage &CopyToColumns(double sparse_threshold) {
    if (column_matrix.GetNumFeature() != 0) {
      return *this;
    }
    this->column_matrix.Init(*this, sparse_threshold);
    return *this;
  }

  // get i-th row
  common::GHistIndexRow operator[](size_t i) const {
    return {&index[0] + row_ptr[i],
            static_cast<common::GHistIndexRow::index_type>(
                row_ptr[i + 1] - row_ptr[i])};
  }

  bst_feature_t NumFeatures() const { return cut.MinValues().size(); }
  bst_row_t Size() const { return row_ptr.size() - 1; }

  float GetFvalue(bst_row_t ridx, bst_feature_t fid) const;
  void GetFeatureCounts(size_t* counts) const {
    auto nfeature = cut.Ptrs().size() - 1;
    for (unsigned fid = 0; fid < nfeature; ++fid) {
      auto ibegin = cut.Ptrs()[fid];
      auto iend = cut.Ptrs()[fid + 1];
      for (auto i = ibegin; i < iend; ++i) {
        counts[fid] += hit_count[i];
      }
    }
  }

 private:
  std::vector<size_t> hit_count_tloc_;
};

class GradientIndexSource : public DataSource<GradientIndexPage> {
  GradientIndexPage gradient_index_;
  bool at_first_;

 public:
  explicit GradientIndexSource(DMatrix* m, const BatchParam& param);

  bool Next() override {
    if (!at_first_) {
      return false;
    }
    at_first_ = false;
    return true;
  }
  void BeforeFirst() override {
    at_first_ = true;
  }
  // implement Value
  GradientIndexPage const& Value() const override {
    return gradient_index_;
  }
  GradientIndexPage& Value() {
    return gradient_index_;
  }
};

}      // namespace xgboost
#endif  // XGBOOST_DATA_GRADIENT_INDEX_SOURCE_H_
