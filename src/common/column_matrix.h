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

namespace xgboost {
namespace common {


/*! \brief column type */
enum ColumnType {
  kDenseColumn,
  kSparseColumn
};

/*! \brief a column storage, to be used with ApplySplit. Note that each
    bin id is stored as index[i] + index_base.
    Different types of column index for each column allow
    to reduce the memory usage. */
template <typename T>
class Column {
 public:
  Column(ColumnType type, const T* index, uint32_t index_base,
         const size_t* row_ind, size_t len,
         const std::vector<bool>& missing_flags,
         const size_t missing_flags_offset)
      : type_(type),
        index_(index),
        index_base_(index_base),
        row_ind_(row_ind),
        len_(len),
        missing_flags_(missing_flags),
        missing_flags_offset_(missing_flags_offset) {}
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
  bool IsMissing(size_t idx) const {
    return missing_flags_[missing_flags_offset_ + idx];
  }
  const size_t* GetRowData() const { return row_ind_; }

  const std::vector<bool>& missing_flags_;
  const size_t missing_flags_offset_;

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
    feature_offsets_.resize(nfeature + 1);
    size_t accum_index_ = 0;
    feature_offsets_[0] = accum_index_;
    for (int32_t fid = 1; fid < nfeature + 1; ++fid) {
      if (type_[fid - 1] == kDenseColumn) {
        accum_index_ += static_cast<size_t>(nrow);
      } else {
        accum_index_ += feature_counts_[fid - 1];
      }
      feature_offsets_[fid] = accum_index_;
    }

    SetTypeSize(gmat.max_num_bins_);

    index_.resize(feature_offsets_[nfeature] * type_size_, 0);
    if (!all_dense) {
      row_ind_.resize(feature_offsets_[nfeature]);
    }

    // store least bin id for each feature
    index_base_ = const_cast<uint32_t*>(gmat.cut.Ptrs().data());

    // pre-fill index_ for dense columns
    const bool noMissingValues = gmat.row_ptr[nrow] == nrow * nfeature;

    if (noMissingValues) {
      missing_flags_.resize(feature_offsets_[nfeature], false);
    } else {
      missing_flags_.resize(feature_offsets_[nfeature], true);
    }

    if (all_dense) {
      switch (gmat.index.getBinBound()) {
        case UINT8_BINS_TYPE:
          SetIndexAllDense(gmat.index.data<uint8_t>(), gmat, nrow, nfeature, noMissingValues);
          break;
        case UINT16_BINS_TYPE:
          SetIndexAllDense(gmat.index.data<uint16_t>(), gmat, nrow, nfeature, noMissingValues);
          break;
        case UINT32_BINS_TYPE:
          SetIndexAllDense(gmat.index.data<uint32_t>(), gmat, nrow, nfeature, noMissingValues);
          break;
        default:
            BinBounds curent_bound = gmat.index.getBinBound();
            CHECK(curent_bound == UINT8_BINS_TYPE ||
                  curent_bound == UINT16_BINS_TYPE ||
                  curent_bound == UINT32_BINS_TYPE);
      }
    /* For sparse DMatrix gmat.index.getBinBound() returns always UINT32_BINS_TYPE
       but for ColumnMatrix we still have a chance to reduce the memory consumption */
    } else {
      switch (type_size_) {
        case sizeof(uint8_t):
          SetIndex<uint8_t>(gmat.index.data<uint32_t>(), gmat, nrow, nfeature);
          break;
        case sizeof(uint16_t):
          SetIndex<uint16_t>(gmat.index.data<uint32_t>(), gmat, nrow, nfeature);
          break;
        case sizeof(uint32_t):
          SetIndex<uint32_t>(gmat.index.data<uint32_t>(), gmat, nrow, nfeature);
          break;
      default:
        CHECK(type_size_ == UINT8_BINS_TYPE ||
              type_size_ == UINT16_BINS_TYPE ||
              type_size_ == UINT32_BINS_TYPE);
      }
    }
  }

  /* Set the number of bytes based on numeric limit of maximum number of bins provided by user */
  void SetTypeSize(size_t max_num_bins) {
    if ( (max_num_bins - 1) <= static_cast<int>(std::numeric_limits<uint8_t>::max()) ) {
      type_size_ = sizeof(uint8_t);
    } else if ((max_num_bins - 1) <= static_cast<int>(std::numeric_limits<uint16_t>::max())) {
      type_size_ = sizeof(uint16_t);
    } else {
      type_size_ = sizeof(uint32_t);
    }
  }

  /* Fetch an individual column. This code should be used with XGBOOST_TYPE_SWITCH
     to determine type of bin id's */
  template <typename T>
  inline Column<T> GetColumn(unsigned fid) const {
    Column<T> c(type_[fid],
                reinterpret_cast<const T*>(&index_[feature_offsets_[fid] * type_size_]),
                index_base_[fid], (type_[fid] == ColumnType::kSparseColumn ?
                &row_ind_[feature_offsets_[fid]] : nullptr),
                feature_offsets_[fid + 1] - feature_offsets_[fid],
                missing_flags_, feature_offsets_[fid]);
    return c;
  }

  template<typename T>
  inline void SetIndexAllDense(T* index, const GHistIndexMatrix& gmat,  const size_t nrow,
                               const size_t nfeature,  const bool noMissingValues) {
    T* local_index = reinterpret_cast<T*>(&index_[0]);

    /* missing values make sense only for column with type kDenseColumn,
       and if no missing values were observed it could be handled much faster. */
    if (noMissingValues) {
      const int32_t nthread = omp_get_max_threads();
      #pragma omp parallel for num_threads(nthread)
      for (omp_ulong rid = 0; rid < nrow; ++rid) {
        const size_t ibegin = rid*nfeature;
        const size_t iend = (rid+1)*nfeature;
        size_t j = 0;
        for (size_t i = ibegin; i < iend; ++i, ++j) {
            const size_t idx = feature_offsets_[j];
            local_index[idx + rid] = index[i];
        }
      }
    } else {
      /* to handle rows in all batches, sum of all batch sizes equal to gmat.row_ptr.size() - 1 */
      size_t rbegin = 0;
      for (const auto &batch : gmat.p_fmat_->GetBatches<SparsePage>()) {
        const xgboost::Entry* data_ptr = batch.data.HostVector().data();
        const std::vector<bst_row_t>& offset_vec = batch.offset.HostVector();
        const size_t batch_size = batch.Size();
        CHECK_LT(batch_size, offset_vec.size());
        for (size_t rid = 0; rid < batch_size; ++rid) {
          const size_t size = offset_vec[rid + 1] - offset_vec[rid];
          SparsePage::Inst inst = {data_ptr + offset_vec[rid], size};
          const size_t ibegin = gmat.row_ptr[rbegin + rid];
          const size_t iend = gmat.row_ptr[rbegin + rid + 1];
          CHECK_EQ(ibegin + inst.size(), iend);
          size_t j = 0;
          size_t fid = 0;
          for (size_t i = ibegin; i < iend; ++i, ++j) {
              fid = inst[j].index;
              const size_t idx = feature_offsets_[fid];
              /* rbegin allows to store indexes from specific SparsePage batch */
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
      const xgboost::Entry* data_ptr = batch.data.HostVector().data();
      const std::vector<bst_row_t>& offset_vec = batch.offset.HostVector();
      const size_t batch_size = batch.Size();
      CHECK_LT(batch_size, offset_vec.size());
      for (size_t rid = 0; rid < batch_size; ++rid) {
        const size_t ibegin = gmat.row_ptr[rbegin + rid];
        const size_t iend = gmat.row_ptr[rbegin + rid + 1];
        size_t fid = 0;
        const size_t size = offset_vec[rid + 1] - offset_vec[rid];
        SparsePage::Inst inst = {data_ptr + offset_vec[rid], size};

        CHECK_EQ(ibegin + inst.size(), iend);
        size_t j = 0;
        for (size_t i = ibegin; i < iend; ++i, ++j) {
          const uint32_t bin_id = index[i];

          fid = inst[j].index;
          if (type_[fid] == kDenseColumn) {
            T* begin = &local_index[feature_offsets_[fid]];
            begin[rid + rbegin] = bin_id - index_base_[fid];
            missing_flags_[feature_offsets_[fid] + rid + rbegin] = false;
          } else {
            T* begin = &local_index[feature_offsets_[fid]];
            begin[num_nonzeros[fid]] = bin_id - index_base_[fid];
            row_ind_[feature_offsets_[fid] + num_nonzeros[fid]] = rid + rbegin;
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

  std::vector<size_t> feature_counts_;
  std::vector<ColumnType> type_;
  std::vector<size_t> row_ind_;
  /* indicate where each column's index and row_ind is stored. */
  std::vector<size_t> feature_offsets_;

  // index_base_[fid]: least bin id for feature fid
  uint32_t* index_base_;
  std::vector<bool> missing_flags_;
  uint32_t type_size_;
};

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_H_
