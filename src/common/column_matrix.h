/*!
 * Copyright 2017 by Contributors
 * \file column_matrix.h
 * \brief Utility for fast column-wise access
 * \author Philip Cho
 */

#define XGBOOST_TYPE_SWITCH(dtype, OP)					\
switch (dtype) {						\
  case xgboost::common::kUInt32 : {						\
    using DType = uint32_t;					\
    OP; break;							\
  }								\
  case xgboost::common::kUInt16 : {						\
    using DType = uint16_t;					\
    OP; break;							\
  }								\
  case xgboost::common::kUInt8 : {						\
    using DType = uint8_t;					\
    OP; break;							\
    default: LOG(FATAL) << "don't recognize type flag" << dtype;	\
  } \
}

#ifndef XGBOOST_COLUMN_MATRIX_H_
#define XGBOOST_COLUMN_MATRIX_H_

#include "hist_util.h"

namespace xgboost {
namespace common {

enum DataType {
  kUInt8 = 1,
  kUInt16 = 2,
  kUInt32 = 4
};

enum ColumnType {
  kDenseColumn,
  kSparseColumn
};

template<typename T>
class Column {
 public:
  ColumnType type_;
  const T* index_;
  uint32_t index_base_;
  const uint32_t* row_ind_;
  size_t len_;
};

class ColumnMatrix {
 public:
  inline uint32_t GetNumFeature() const {
    return type_.size();
  }

  inline void Init(const GHistIndexMatrix& gmat, DataType dtype) {
    this->dtype = dtype;
    packing_factor_ = sizeof(uint32_t) / static_cast<size_t>(this->dtype);

    const uint32_t nfeature = gmat.cut->row_ptr.size() - 1;
    const omp_ulong nrow = static_cast<omp_ulong>(gmat.row_ptr.size() - 1);
    int nthread;
    #pragma omp parallel
    {
      nthread = omp_get_num_threads();
    }
    nthread = std::max(nthread / 2, 1);

    // identify type of each column
    feature_counts_.resize(nfeature);
    type_.resize(nfeature);
    std::fill(feature_counts_.begin(), feature_counts_.end(), 0);

    uint32_t max_val = 0;
    XGBOOST_TYPE_SWITCH(this->dtype, {
      max_val = std::numeric_limits<DType>::max();
    });
    for (uint32_t fid = 0; fid < nfeature; ++fid) {
      CHECK_LE(gmat.cut->row_ptr[fid + 1] - gmat.cut->row_ptr[fid], max_val);
    }

    gmat.GetFeatureCounts(&feature_counts_[0]);
    // classify features
    for (uint32_t fid = 0; fid < nfeature; ++fid) {
      if (static_cast<double>(feature_counts_[fid]) < 0.5*nrow) {
        type_[fid] = kSparseColumn;
      } else {
        type_[fid] = kDenseColumn;
      }
    }

    // want to compute storage boundary for each feature
    // using variants of prefix sum scan
    boundary_.resize(nfeature);
    bst_uint accum_index_ = 0;
    bst_uint accum_row_ind_ = 0;
    for (uint32_t fid = 0; fid < nfeature; ++fid) {
      boundary_[fid].index_begin = accum_index_;
      boundary_[fid].row_ind_begin = accum_row_ind_;
      if (type_[fid] == kDenseColumn) {
        accum_index_ += nrow;
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
    for (uint32_t fid = 0; fid < nfeature; ++fid) {
      index_base_[fid] = gmat.cut->row_ptr[fid];
    }

    // fill index_ for dense columns
    for (uint32_t fid = 0; fid < nfeature; ++fid) {
      if (type_[fid] == kDenseColumn) {
        const uint32_t ibegin = boundary_[fid].index_begin;
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
    // top_ind[fid] = how many nonzeros have this feature accumulated so far?
    std::vector<uint32_t> top_ind;
    top_ind.resize(nfeature);
    std::fill(top_ind.begin(), top_ind.end(), 0);
    for (uint32_t rid = 0; rid < nrow; ++rid) {
      const size_t ibegin = static_cast<size_t>(gmat.row_ptr[rid]);
      const size_t iend = static_cast<size_t>(gmat.row_ptr[rid + 1]);
      size_t fid = 0;
      for (size_t i = ibegin; i < iend; ++i) {
        const size_t bin_id = gmat.index[i];
        while (bin_id >= gmat.cut->row_ptr[fid + 1]) {
          ++fid;
        }
        if (type_[fid] == kDenseColumn) {
          XGBOOST_TYPE_SWITCH(this->dtype, {
            const size_t block_offset = boundary_[fid].index_begin / packing_factor_;
            const size_t elem_offset = boundary_[fid].index_begin % packing_factor_;
            DType* begin = reinterpret_cast<DType*>(&index_[block_offset]) + elem_offset;
            begin[rid] = bin_id - index_base_[fid];
          });
        } else {
          XGBOOST_TYPE_SWITCH(this->dtype, {
            const size_t block_offset = boundary_[fid].index_begin / packing_factor_;
            const size_t elem_offset = boundary_[fid].index_begin % packing_factor_;
            DType* begin = reinterpret_cast<DType*>(&index_[block_offset]) + elem_offset;
            begin[top_ind[fid]] = bin_id - index_base_[fid];
          });
          row_ind_[boundary_[fid].row_ind_begin + top_ind[fid]] = rid;
          ++top_ind[fid];
        }
      }
    }
  }

  template<typename T>
  inline Column<T> GetColumn(unsigned fid) const {
    Column<T> c;

    c.type_ = type_[fid];
    const size_t block_offset = boundary_[fid].index_begin / packing_factor_;
    const size_t elem_offset = boundary_[fid].index_begin % packing_factor_;
    c.index_ = reinterpret_cast<const T*>(&index_[block_offset]) + elem_offset;
    c.index_base_ = index_base_[fid];
    c.row_ind_ = &row_ind_[boundary_[fid].row_ind_begin];
    c.len_ = boundary_[fid].index_end - boundary_[fid].index_begin;

    return c;
  }

 public:
  DataType dtype;

 private:
  struct ColumnBoundary {
    unsigned index_begin;
    unsigned index_end;
    unsigned row_ind_begin;
    unsigned row_ind_end;
  };

  std::vector<bst_uint> feature_counts_;
  std::vector<ColumnType> type_;
  std::vector<uint32_t> index_;  // index_: may store smaller integers; needs padding
  std::vector<uint32_t> row_ind_;
  std::vector<ColumnBoundary> boundary_;

  size_t packing_factor_;

  // index_base_[fid]: least bin id for feature fid
  std::vector<uint32_t> index_base_;
};

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COLUMN_MATRIX_H_
