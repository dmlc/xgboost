/*!
 * Copyright 2017 by Contributors
 * \file column_matrix.h
 * \brief Utility for fast column-wise access
 * \author Philip Cho
 */

#ifndef XGBOOST_COMMON_COLUMN_MATRIX_H_
#define XGBOOST_COMMON_COLUMN_MATRIX_H_

#include <limits>
#include <vector>
#include "hist_util.h"

uint64_t get_time();

namespace xgboost {
namespace common {


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
         const size_t* row_ind, size_t len,
         const std::vector<bool>* missing_flags, const size_t disp)
      : type_(type),
        index_(index),
        index_base_(index_base),
        row_ind_(row_ind),
        len_(len),
        missing_flags_(missing_flags), disp_(disp) {}
  size_t Size() const { return len_; }
  uint32_t GetGlobalBinIdx(size_t idx) const { return index_base_ + (uint32_t)(index_[idx]); }
  T GetFeatureBinIdx(size_t idx) const { return index_[idx]; }
  common::Span<const T> GetFeatureBinIdxPtr() const { return { index_, len_ }; }
  // column.GetFeatureBinIdx(idx) + column.GetBaseIdx(idx) ==
  // column.GetGlobalBinIdx(idx)
  uint32_t GetBaseIdx() const { return index_base_; }
  ColumnType GetType() const { return type_; }
  size_t GetRowIdx(size_t idx) const {
    // clang-tidy worries that row_ind_ might be a nullptr, which is possible,
    // but low level structure is not safe anyway.
    return type_ == ColumnType::kDenseColumn ? idx : row_ind_[idx];// NOLINT
  }
  inline bool IsMissing(size_t idx) const {
    return (*missing_flags_)[disp_ + idx] == true;
  }
  const size_t* GetRowData() const { return row_ind_; }

  const std::vector<bool>* missing_flags_;
  const size_t disp_;

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
                   double  sparse_threshold) {
    const int32_t nfeature = static_cast<int32_t>(gmat.cut.Ptrs().size() - 1);
    const size_t nrow = gmat.row_ptr.size() - 1;
    // identify type of each column
    feature_counts_.resize(nfeature);
    type_.resize(nfeature);
    std::fill(feature_counts_.begin(), feature_counts_.end(), 0);

    uint32_t max_val = std::numeric_limits<uint32_t>::max();
    for (int32_t fid = 0; fid < nfeature; ++fid) {
      CHECK_LE(gmat.cut.Ptrs()[fid + 1] - gmat.cut.Ptrs()[fid], max_val);
    }
    bool all_dense = gmat.IsDense();
    gmat.GetFeatureCounts(&feature_counts_[0]);
    // classify features
    for (int32_t fid = 0; fid < nfeature; ++fid) {
      if (static_cast<double>(feature_counts_[fid])
                 < sparse_threshold * nrow) {
        type_[fid] = kSparseColumn;
        all_dense = false;
      } else {
        type_[fid] = kDenseColumn;
      }
    }

    // want to compute storage boundary for each feature
    // using variants of prefix sum scan
    boundary_.resize(nfeature);
    size_t accum_index_ = 0;
    size_t accum_row_ind_ = 0;
    for (int32_t fid = 0; fid < nfeature; ++fid) {
      boundary_[fid].index_begin = accum_index_;
      if (type_[fid] == kDenseColumn) {
        accum_index_ += static_cast<size_t>(nrow);
      } else {
        accum_index_ += feature_counts_[fid];
        accum_row_ind_ += feature_counts_[fid];
      }
      boundary_[fid].index_end = accum_index_;
    }

    if ( (gmat.max_num_bins_ - 1) <= static_cast<int>(std::numeric_limits<uint8_t>::max()) ) {
      type_size_ = 1;
    } else if ((gmat.max_num_bins_ - 1) <= static_cast<int>(std::numeric_limits<uint16_t>::max())) {
      type_size_ = 2;
    } else {
      type_size_ = 4;
    }

    index_.resize(boundary_[nfeature - 1].index_end * type_size_, 0);
    if (!all_dense) {
      row_ind_.resize(boundary_[nfeature - 1].index_end);
    }

    // store least bin id for each feature
    index_base_ = const_cast<uint32_t*>(gmat.cut.Ptrs().data());

    // pre-fill index_ for dense columns
    const bool noMissingValues = gmat.row_ptr[nrow] == nrow * nfeature;

    if (noMissingValues) {
      missing_flags_.resize(boundary_[nfeature - 1].index_end, false);
    } else {
      missing_flags_.resize(boundary_[nfeature - 1].index_end, true);
    }

    if (all_dense) {
      switch (gmat.index.getBinBound()) {
        case POWER_OF_TWO_8:
          SetIndexAllDense(gmat.index.data<uint8_t>(), gmat, nrow, nfeature, noMissingValues);
          break;
        case POWER_OF_TWO_16:
          SetIndexAllDense(gmat.index.data<uint16_t>(), gmat, nrow, nfeature, noMissingValues);
          break;
        case POWER_OF_TWO_32:
          SetIndexAllDense(gmat.index.data<uint32_t>(), gmat, nrow, nfeature, noMissingValues);
          break;
      }
    } else {
      switch (type_size_) {
        case 1:
          SetIndex<uint8_t>(gmat.index.data<uint32_t>(), gmat, nrow, nfeature);
          break;
        case 2:
          SetIndex<uint16_t>(gmat.index.data<uint32_t>(), gmat, nrow, nfeature);
          break;
        case 4:
          SetIndex<uint32_t>(gmat.index.data<uint32_t>(), gmat, nrow, nfeature);
          break;
      }
    }
  }

  /* Fetch an individual column. This code should be used with XGBOOST_TYPE_SWITCH
     to determine type of bin id's */
  template <typename T>
  inline Column<T> GetColumn(unsigned fid) const {
    Column<T> c(type_[fid],
                reinterpret_cast<const T*>(&index_[boundary_[fid].index_begin * type_size_]),
                index_base_[fid], (type_[fid] == ColumnType::kSparseColumn ?
                &row_ind_[boundary_[fid].index_begin] : nullptr),
                boundary_[fid].index_end - boundary_[fid].index_begin,
                &missing_flags_, boundary_[fid].index_begin);
    return c;
  }

  template<typename T>
  inline void SetIndexAllDense(T* index, const GHistIndexMatrix& gmat,  const size_t nrow,
                               const size_t nfeature,  const bool noMissingValues) {
    T* local_index = reinterpret_cast<T*>(&index_[0]);

    if (noMissingValues) {
      const int32_t nthread = omp_get_max_threads();
      #pragma omp parallel for num_threads(nthread)
      for (omp_ulong rid = 0; rid < nrow; ++rid) {
        const size_t ibegin = rid*nfeature;
        const size_t iend = (rid+1)*nfeature;
        size_t j = 0;
        for (size_t i = ibegin; i < iend; ++i, ++j) {
            const size_t idx = boundary_[j].index_begin;
            local_index[idx + rid] = index[i];
        }
      }
    } else {
      size_t rbegin = 0;
      for (const auto &batch : gmat.p_fmat_->GetBatches<SparsePage>()) {
        for (size_t rid = 0; rid < batch.Size(); ++rid) {
          SparsePage::Inst inst = batch[rid];
          const size_t ibegin = gmat.row_ptr[rbegin + rid];
          const size_t iend = gmat.row_ptr[rbegin + rid + 1];
          CHECK_EQ(ibegin + inst.size(), iend);
          size_t j = 0;
          size_t fid = 0;
          for (size_t i = ibegin; i < iend; ++i, ++j) {
              fid = inst[j].index;
              const size_t idx = boundary_[fid].index_begin;
              local_index[idx + rbegin + rid] = index[i];
              missing_flags_[idx + rbegin + rid] = false;
          }
        }
        rbegin += batch.Size();
      }
    }
  }

  template<typename T>
  inline void SetIndex(uint32_t* index, const GHistIndexMatrix& gmat,
                       const size_t nrow, const size_t nfeature) {
    std::vector<size_t> num_nonzeros;
    num_nonzeros.resize(nfeature);
    std::fill(num_nonzeros.begin(), num_nonzeros.end(), 0);

    T* local_index = reinterpret_cast<T*>(&index_[0]);

    size_t rbegin = 0;
    for (const auto &batch : gmat.p_fmat_->GetBatches<SparsePage>()) {
      for (size_t rid = 0; rid < batch.Size(); ++rid) {
        const size_t ibegin = gmat.row_ptr[rbegin + rid];
        const size_t iend = gmat.row_ptr[rbegin + rid + 1];
        size_t fid = 0;
        SparsePage::Inst inst = batch[rid];
        CHECK_EQ(ibegin + inst.size(), iend);
        size_t jp = 0;
        for (size_t i = ibegin; i < iend; ++i, ++jp) {
          const uint32_t bin_id = index[i];

          fid = inst[jp].index;
          if (type_[fid] == kDenseColumn) {
            T* begin = &local_index[boundary_[fid].index_begin];
            begin[rid + rbegin] = bin_id - index_base_[fid];
            missing_flags_[boundary_[fid].index_begin + rid + rbegin] = false;
          } else {
            T* begin = &local_index[boundary_[fid].index_begin];
            begin[num_nonzeros[fid]] = bin_id - index_base_[fid];
            row_ind_[boundary_[fid].index_begin + num_nonzeros[fid]] = rid + rbegin;
            ++num_nonzeros[fid];
          }
        }
      }
      rbegin += batch.Size();
    }
  }
  const size_t GetTypeSize() const {
    return type_size_;
  }

 private:
  std::vector<uint8_t> index_;  // index_: may store smaller integers; needs padding
  struct ColumnBoundary {
    // indicate where each column's index and row_ind is stored.
    // index_begin and index_end are logical offsets, so they should be converted to
    // actual offsets by scaling with packing_factor_
    size_t index_begin;
    size_t index_end;
  };

  std::vector<size_t> feature_counts_;
  std::vector<ColumnType> type_;
  std::vector<size_t> row_ind_;
  std::vector<ColumnBoundary> boundary_;

  // index_base_[fid]: least bin id for feature fid
  uint32_t* index_base_;
  std::vector<bool> missing_flags_;
  uint32_t type_size_;
};

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_H_
