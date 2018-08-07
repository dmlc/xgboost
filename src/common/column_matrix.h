/*!
 * Copyright 2017 by Contributors
 * \file column_matrix.h
 * \brief Utility for fast column-wise access
 * \author Philip Cho
 */

#ifndef XGBOOST_COMMON_COLUMN_MATRIX_H_
#define XGBOOST_COMMON_COLUMN_MATRIX_H_

#define XGBOOST_TYPE_SWITCH(dtype, OP)                      \
  \
switch(dtype) {                                             \
    case xgboost::common::uint32: {                         \
      using DType = uint32_t;                               \
      OP;                                                   \
      break;                                                \
    }                                                       \
    case xgboost::common::uint16: {                         \
      using DType = uint16_t;                               \
      OP;                                                   \
      break;                                                \
    }                                                       \
    case xgboost::common::uint8: {                          \
      using DType = uint8_t;                                \
      OP;                                                   \
      break;                                                \
      default:                                              \
        LOG(FATAL) << "don't recognize type flag" << dtype; \
    }                                                       \
  \
}

#include <type_traits>
#include <limits>
#include <vector>
#include "hist_util.h"


namespace xgboost {
namespace common {

/*! \brief indicator of data type used for storing bin id's in a column. */
enum BinIdxStorageType {
  uint8 = 1,
  uint16 = 2,
  uint32 = 4
};

/*! \brief column type */
enum ColumnType {
  kDenseColumn,
  kSparseColumn
};

/*! \brief a column storage, to be used with ApplySplit. Note that each
    bin id is stored as index[i] + index_base. */
template <typename T>
class Column {
 public:
  Column(ColumnType type, const T* index, uint32_t index_base,
         const size_t* row_ind, size_t len)
      : type_(type),
        index_(index),
        index_base_(index_base),
        row_ind_(row_ind),
        len_(len) {}
  size_t Size() const { return len_; }
  T GetGlobalBinIdx(size_t idx) const { return index_base_ + index_[idx]; }
  T GetFeatureBinIdx(size_t idx) const { return index_[idx]; }
  // column.GetFeatureBinIdx(idx) + column.GetBaseIdx(idx) ==
  // column.GetGlobalBinIdx(idx)
  uint32_t GetBaseIdx() const { return index_base_; }
  ColumnType GetType() const { return type_; }
  size_t GetRowIdx(size_t idx) const {
    return type_ == ColumnType::kDenseColumn ? idx : row_ind_[idx];
  }
  bool IsMissing(size_t idx) const {
    return index_[idx] == std::numeric_limits<T>::max();
  }
  const size_t* GetRowData() const { return row_ind_; }

 private:
  ColumnType type_;
  const T* index_;
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
  inline void Init(const GHistIndexMatrix& gmat,
                BinIdxStorageType storage_type, double  sparse_threshold) {
    this->dtype = storage_type;
    /* if dtype is smaller than uint32_t, multiple bin_id's will be stored in each
       slot of internal buffer. */
    packing_factor_ = sizeof(uint32_t) / static_cast<size_t>(this->dtype);

    const auto nfeature = static_cast<bst_uint>(gmat.cut.row_ptr.size() - 1);
    const size_t nrow = gmat.row_ptr.size() - 1;

    // identify type of each column
    feature_counts_.resize(nfeature);
    type_.resize(nfeature);
    std::fill(feature_counts_.begin(), feature_counts_.end(), 0);

    uint32_t max_val = 0;
    XGBOOST_TYPE_SWITCH(this->dtype, {
      max_val = static_cast<uint32_t>(std::numeric_limits<DType>::max());
    });
    for (bst_uint fid = 0; fid < nfeature; ++fid) {
      CHECK_LE(gmat.cut.row_ptr[fid + 1] - gmat.cut.row_ptr[fid], max_val);
    }

    gmat.GetFeatureCounts(&feature_counts_[0]);
    // classify features
    for (bst_uint fid = 0; fid < nfeature; ++fid) {
      if (static_cast<double>(feature_counts_[fid])
                 < sparse_threshold * nrow) {
        type_[fid] = kSparseColumn;
      } else {
        type_[fid] = kDenseColumn;
      }
    }

    // want to compute storage boundary for each feature
    // using variants of prefix sum scan
    boundary_.resize(nfeature);
    size_t accum_index_ = 0;
    size_t accum_row_ind_ = 0;
    for (bst_uint fid = 0; fid < nfeature; ++fid) {
      boundary_[fid].index_begin = accum_index_;
      boundary_[fid].row_ind_begin = accum_row_ind_;
      if (type_[fid] == kDenseColumn) {
        accum_index_ += static_cast<size_t>(nrow);
        accum_row_ind_ += static_cast<size_t>(nrow);
      } else {
        accum_index_ += feature_counts_[fid];
        accum_row_ind_ += feature_counts_[fid];
      }
      boundary_[fid].index_end = accum_index_;
      boundary_[fid].row_ind_end = accum_row_ind_;
    }

    index_.resize((boundary_[nfeature - 1].index_end
                   + (packing_factor_ - 1)) / packing_factor_);
    row_ind_.resize(boundary_[nfeature - 1].row_ind_end);

    // store least bin id for each feature
    index_base_.resize(nfeature);
    for (bst_uint fid = 0; fid < nfeature; ++fid) {
      index_base_[fid] = gmat.cut.row_ptr[fid];
    }

    // pre-fill index_ for dense columns
    for (bst_uint fid = 0; fid < nfeature; ++fid) {
      if (type_[fid] == kDenseColumn) {
        const size_t ibegin = boundary_[fid].index_begin;
        XGBOOST_TYPE_SWITCH(this->dtype, {
          const size_t block_offset = ibegin / packing_factor_;
          const size_t elem_offset = ibegin % packing_factor_;
          DType* begin = reinterpret_cast<DType*>(&index_[block_offset]) + elem_offset;
          DType* end = begin + nrow;
          std::fill(begin, end, std::numeric_limits<DType>::max());
            // max() indicates missing values
        });
      }
    }

    // loop over all rows and fill column entries
    // num_nonzeros[fid] = how many nonzeros have this feature accumulated so far?
    std::vector<size_t> num_nonzeros;
    num_nonzeros.resize(nfeature);
    std::fill(num_nonzeros.begin(), num_nonzeros.end(), 0);
    for (size_t rid = 0; rid < nrow; ++rid) {
      const size_t ibegin = gmat.row_ptr[rid];
      const size_t iend = gmat.row_ptr[rid + 1];
      size_t fid = 0;
      for (size_t i = ibegin; i < iend; ++i) {
        const uint32_t bin_id = gmat.index[i];
        while (bin_id >= gmat.cut.row_ptr[fid + 1]) {
          ++fid;
        }
        if (type_[fid] == kDenseColumn) {
          XGBOOST_TYPE_SWITCH(this->dtype, {
            const size_t block_offset = boundary_[fid].index_begin / packing_factor_;
            const size_t elem_offset = boundary_[fid].index_begin % packing_factor_;
            DType* begin = reinterpret_cast<DType*>(&index_[block_offset]) + elem_offset;
            begin[rid] = static_cast<DType>(bin_id - index_base_[fid]);
          });
        } else {
          XGBOOST_TYPE_SWITCH(this->dtype, {
            const size_t block_offset = boundary_[fid].index_begin / packing_factor_;
            const size_t elem_offset = boundary_[fid].index_begin % packing_factor_;
            DType* begin = reinterpret_cast<DType*>(&index_[block_offset]) + elem_offset;
            begin[num_nonzeros[fid]] = static_cast<DType>(bin_id - index_base_[fid]);
          });
          row_ind_[boundary_[fid].row_ind_begin + num_nonzeros[fid]] = rid;
          ++num_nonzeros[fid];
        }
      }
    }
  }

  /* Fetch an individual column. This code should be used with XGBOOST_TYPE_SWITCH
     to determine type of bin id's */
  template <typename T>
  inline Column<T> GetColumn(unsigned fid) const {
    const bool valid_type = std::is_same<T, uint32_t>::value ||
                            std::is_same<T, uint16_t>::value ||
                            std::is_same<T, uint8_t>::value;
    CHECK(valid_type);

    const size_t block_offset = boundary_[fid].index_begin / packing_factor_;
    const size_t elem_offset = boundary_[fid].index_begin % packing_factor_;
    Column<T> c(type_[fid],
                reinterpret_cast<const T*>(&index_[block_offset]) + elem_offset,
                index_base_[fid], &row_ind_[boundary_[fid].row_ind_begin],
                boundary_[fid].index_end - boundary_[fid].index_begin);
    return c;
  }

 public:
  BinIdxStorageType dtype;

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

  size_t packing_factor_;  // how many integers are stored in each slot of index_

  // index_base_[fid]: least bin id for feature fid
  std::vector<uint32_t> index_base_;
};

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_H_
