/*!
 * Copyright 2017-2020 by Contributors
 * \file column_matrix_oneapi.h
 * \brief Utility for fast column-wise access
 */

#ifndef XGBOOST_COMMON_COLUMN_MATRIX_ONEAPI_H_
#define XGBOOST_COMMON_COLUMN_MATRIX_ONEAPI_H_

#include <limits>
#include <vector>
#include <memory>
#include "../../src/common/column_matrix.h"
#include "hist_util_oneapi.h"

namespace xgboost {
namespace common {

template <typename BinIdxType>
class ColumnOneAPI {
 public:
  ColumnOneAPI(ColumnType type, common::Span<const BinIdxType> index, const uint32_t index_base)
      : type_(type),
        index_(index),
        index_base_(index_base) {}

  virtual ~ColumnOneAPI() = default;

  uint32_t GetGlobalBinIdx(size_t idx) const {
    return index_base_ + static_cast<uint32_t>(index_[idx]);
  }

  BinIdxType GetFeatureBinIdx(size_t idx) const { return index_[idx]; }

  const uint32_t GetBaseIdx() const { return index_base_; }

  common::Span<const BinIdxType> GetFeatureBinIdxPtr() const { return index_; }

  ColumnType GetType() const { return type_; }

  /* returns number of elements in column */
  size_t Size() const { return index_.size(); }

 private:
  /* type of column */
  ColumnType type_;
  /* bin indexes in range [0, max_bins - 1] */
  common::Span<const BinIdxType> index_;
  /* bin index offset for specific feature */
  const uint32_t index_base_;
};

template <typename BinIdxType>
class SparseColumnOneAPI: public ColumnOneAPI<BinIdxType> {
 public:
  SparseColumnOneAPI(ColumnType type, common::Span<const BinIdxType> index,
              uint32_t index_base, common::Span<const size_t> row_ind)
      : ColumnOneAPI<BinIdxType>(type, index, index_base),
        row_ind_(row_ind) {}

  const size_t* GetRowData() const { return row_ind_.data(); }

  size_t GetRowIdx(size_t idx) const {
    return row_ind_.data()[idx];
  }

 private:
  /* indexes of rows */
  common::Span<const size_t> row_ind_;
};

template <typename BinIdxType>
class DenseColumnOneAPI: public ColumnOneAPI<BinIdxType> {
 public:
  DenseColumnOneAPI(ColumnType type, common::Span<const BinIdxType> index,
              uint32_t index_base, bool* missing_flags,
              size_t feature_offset)
      : ColumnOneAPI<BinIdxType>(type, index, index_base),
        missing_flags_(missing_flags),
        feature_offset_(feature_offset) {}

  bool IsMissing(size_t idx) const { return missing_flags_[feature_offset_ + idx]; }

  bool* GetMissingFlags() const {
    return missing_flags_;
  }

  size_t GetFeatureOffset() const {
    return feature_offset_;
  }

 private:
  /* flags for missing values in dense columns */
  bool* missing_flags_;
  size_t feature_offset_;
};

/*! \brief a collection of columns, with support for construction from
    GHistIndexMatrixOneAPI. */
class ColumnMatrixOneAPI {
 public:
  // get number of features
  inline bst_uint GetNumFeature() const {
    return static_cast<bst_uint>(type_.Size());
  }

  // construct column matrix from GHistIndexMatrixOneAPI
  inline void Init(cl::sycl::queue qu,
                   const GHistIndexMatrixOneAPI& gmat,
                   const DeviceMatrixOneAPI& dmat_device,
                   double sparse_threshold) {
    qu_ = qu;
    const int32_t nfeature = static_cast<int32_t>(gmat.cut.Ptrs().size() - 1);
    const size_t nrow = gmat.row_ptr.size() - 1;
    // identify type of each column
    feature_counts_.Resize(qu_, nfeature);
    type_.Resize(qu_, nfeature);
    std::fill(feature_counts_.Begin(), feature_counts_.End(), 0);
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
        all_dense = false;
      }
      type_[fid] = kDenseColumn;
    }

    if (!all_dense) {
      LOG(WARNING) << "Sparse data not supported for updater_quantil_hist_oneapi, converting to dense";
      all_dense = true;
    }

    // want to compute storage boundary for each feature
    // using variants of prefix sum scan
    feature_offsets_.Resize(qu_, nfeature + 1);
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

    SetTypeSize(gmat.max_num_bins);

    index_.Resize(qu_, feature_offsets_[nfeature] * bins_type_size_, 0);
    if (!all_dense) {
      row_ind_.Resize(qu_, feature_offsets_[nfeature]);
    }

    // store least bin id for each feature
    index_base_ = const_cast<uint32_t*>(gmat.cut.Ptrs().data());

    const bool noMissingValues = NoMissingValues(gmat.row_ptr[nrow], nrow, nfeature);
    any_missing_ = !noMissingValues;

    if (noMissingValues) {
      missing_flags_.Resize(qu_, feature_offsets_[nfeature], false);
    } else {
      missing_flags_.Resize(qu_, feature_offsets_[nfeature], true);
    }

    // pre-fill index_ for dense columns
    if (all_dense) {
      BinTypeSize gmat_bin_size = gmat.index.GetBinTypeSize();
      if (gmat_bin_size == kUint8BinsTypeSize) {
          SetIndexAllDense(gmat.index.data<uint8_t>(), gmat, dmat_device, nrow, nfeature, noMissingValues);
      } else if (gmat_bin_size == kUint16BinsTypeSize) {
          SetIndexAllDense(gmat.index.data<uint16_t>(), gmat, dmat_device, nrow, nfeature, noMissingValues);
      } else {
          CHECK_EQ(gmat_bin_size, kUint32BinsTypeSize);
          SetIndexAllDense(gmat.index.data<uint32_t>(), gmat, dmat_device, nrow, nfeature, noMissingValues);
      }
    }
  }

  /* Set the number of bytes based on numeric limit of maximum number of bins provided by user */
  void SetTypeSize(size_t max_num_bins) {
    if ( (max_num_bins - 1) <= static_cast<int>(std::numeric_limits<uint8_t>::max()) ) {
      bins_type_size_ = kUint8BinsTypeSize;
    } else if ((max_num_bins - 1) <= static_cast<int>(std::numeric_limits<uint16_t>::max())) {
      bins_type_size_ = kUint16BinsTypeSize;
    } else {
      bins_type_size_ = kUint32BinsTypeSize;
    }
  }

  /* Fetch an individual column. This code should be used with type swith
     to determine type of bin id's */
  template <typename BinIdxType>
  std::unique_ptr<const ColumnOneAPI<BinIdxType> > GetColumn(unsigned fid) const {
    CHECK_EQ(sizeof(BinIdxType), bins_type_size_);

    const size_t feature_offset = feature_offsets_[fid];  // to get right place for certain feature
    const size_t column_size = feature_offsets_[fid + 1] - feature_offset;
    common::Span<const BinIdxType> bin_index = { reinterpret_cast<const BinIdxType*>(
                                                 &index_[feature_offset * bins_type_size_]),
                                                 column_size };
    std::unique_ptr<const ColumnOneAPI<BinIdxType> > res;
    if (type_[fid] == ColumnType::kDenseColumn) {
      res.reset(new DenseColumnOneAPI<BinIdxType>(type_[fid], bin_index, index_base_[fid],
                                                  missing_flags_.Begin(), feature_offset));
    } else {
      res.reset(new SparseColumnOneAPI<BinIdxType>(type_[fid], bin_index, index_base_[fid],
                                                   {&row_ind_[feature_offset], column_size}));
    }
    return res;
  }

  template<typename T>
  inline void SetIndexAllDense(const T* index,
                               const GHistIndexMatrixOneAPI& gmat,
                               const DeviceMatrixOneAPI& dmat_device,
                               const size_t nrow,
                               const size_t nfeature,
                               const bool noMissingValues) {
    T* local_index = reinterpret_cast<T*>(&index_[0]);

    const size_t* feature_offsets = feature_offsets_.DataConst(); 

    /* missing values make sense only for column with type kDenseColumn,
       and if no missing values were observed it could be handled much faster. */
    if (noMissingValues) {
      qu_.submit([&](cl::sycl::handler& cgh){
        cgh.parallel_for<>(cl::sycl::range<2>(nrow, nfeature), [=](cl::sycl::item<2> pid) {
          int i = pid.get_id(0);
          int j = pid.get_id(1);
          local_index[i + feature_offsets[j]] = index[i * nfeature + j];
        });
      }).wait();
    } else {
      const xgboost::EntryOneAPI *data_ptr = dmat_device.data.DataConst();
      const bst_row_t *offset_vec = dmat_device.row_ptr.DataConst();
      const size_t num_rows = dmat_device.row_ptr.Size() - 1;
      bool* missing_flags = missing_flags_.Data();

      qu_.submit([&](cl::sycl::handler& cgh) {
        cgh.parallel_for<>(cl::sycl::range<1>(num_rows), [=](cl::sycl::item<1> pid) {
          const size_t i = pid.get_id(0);
          const size_t ibegin = offset_vec[i];
          const size_t iend = offset_vec[i + 1];
          const size_t size = iend - ibegin;
          for (bst_uint j = 0; j < size; ++j) {
            const size_t idx = feature_offsets[data_ptr[ibegin + j].index];
            local_index[i + idx] = index[ibegin + j];
            missing_flags[i + idx] = false;
          }
        });
      }).wait();
    }
  }

  const BinTypeSize GetTypeSize() const {
    return bins_type_size_;
  }

  // This is just an utility function
  const bool NoMissingValues(const size_t n_elements,
                             const size_t n_row, const size_t n_features) {
    return n_elements == n_features * n_row;
  }

  // And this returns part of state
  const bool AnyMissing() const {
    return any_missing_;
  }

 private:
  USMVector<uint8_t> index_;

  USMVector<size_t> feature_counts_;
  USMVector<ColumnType> type_;
  USMVector<size_t> row_ind_;
  USMVector<size_t> feature_offsets_;

  uint32_t* index_base_;
  USMVector<bool> missing_flags_;
  BinTypeSize bins_type_size_;
  bool any_missing_;

  cl::sycl::queue qu_;
};

}  // namespace common
}  // namespace xgboost
#endif  // XGBOOST_COMMON_COLUMN_MATRIX_ONEAPI_H_